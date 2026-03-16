use lsp_server::{Connection, Message, Request, Response};
use lsp_types::{
    DiagnosticOptions, DiagnosticServerCapabilities, DocumentDiagnosticParams,
    DocumentDiagnosticReport, FullDocumentDiagnosticReport, Hover, HoverContents, HoverParams,
    HoverProviderCapability, MarkupContent, MarkupKind, PublishDiagnosticsParams,
    RelatedFullDocumentDiagnosticReport, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Url,
    notification::PublishDiagnostics,
    request::{DocumentDiagnosticRequest, HoverRequest},
};
#[cfg(test)]
use shapels::analyze_source;
use shapels::{ModuleCache, analyze_source_at_path_with_cache, analyze_source_with_cache};
use std::collections::HashMap;

mod cli;
#[cfg(test)]
mod tests;

use cli::{parse_args, run_analysis_if_args};

fn main() {
    let cli_args = parse_args();
    // command mode if args where provided: exits early
    run_analysis_if_args(cli_args);

    let (connection, io_threads) = Connection::stdio();

    let server_capabilities = serde_json::to_value(ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        diagnostic_provider: Some(DiagnosticServerCapabilities::Options(DiagnosticOptions {
            work_done_progress_options: Default::default(),
            identifier: None,
            inter_file_dependencies: false,
            workspace_diagnostics: false,
        })),
        ..Default::default()
    })
    .unwrap();

    let _init_params = connection.initialize(server_capabilities).unwrap();

    let mut documents: HashMap<Url, String> = HashMap::new();
    let mut module_cache = ModuleCache::new();

    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req).unwrap() {
                    break;
                }
                handle_request(&req, &connection, &mut documents, &mut module_cache);
            }
            Message::Notification(notif) => {
                match notif.method.as_str() {
                    "textDocument/didOpen" => {
                        if let Ok(params) = serde_json::from_value::<
                            lsp_types::DidOpenTextDocumentParams,
                        >(notif.params.clone())
                        {
                            let uri = params.text_document.uri.clone();
                            let text = params.text_document.text;
                            if let Ok(path) = uri.to_file_path() {
                                module_cache.update_file_source(&path, text.clone());
                            }
                            documents.insert(uri.clone(), text);
                            publish_diagnostics(&connection, &documents, &uri, &mut module_cache);
                        }
                    }
                    "textDocument/didChange" => {
                        if let Ok(params) = serde_json::from_value::<
                            lsp_types::DidChangeTextDocumentParams,
                        >(notif.params.clone())
                            && let Some(first) = params.content_changes.first()
                        {
                            let uri = params.text_document.uri.clone();
                            // assuming full sync kind
                            if let Ok(path) = uri.to_file_path() {
                                module_cache.update_file_source(&path, first.text.clone());
                            }
                            documents.insert(uri.clone(), first.text.clone());
                            publish_diagnostics(&connection, &documents, &uri, &mut module_cache);
                        }
                    }
                    _ => {}
                }
            }
            Message::Response(_) => {}
        }
    }

    io_threads.join().expect("Failed to join IO threads");
}

fn handle_request(
    req: &Request,
    connection: &Connection,
    documents: &mut HashMap<Url, String>,
    module_cache: &mut ModuleCache,
) {
    match req.method.as_str() {
        <HoverRequest as lsp_types::request::Request>::METHOD => {
            let id = req.id.clone();
            let params: HoverParams = serde_json::from_value(req.params.clone()).unwrap();
            let uri = params.text_document_position_params.text_document.uri;
            let pos = params.text_document_position_params.position;
            let result = documents.get(&uri).and_then(|text| {
                let analysis = uri
                    .to_file_path()
                    .ok()
                    .map(|p| analyze_source_at_path_with_cache(text, &p, module_cache))
                    .unwrap_or_else(|| analyze_source_with_cache(text, module_cache));
                analysis.hover(pos).and_then(|info| {
                    info.shape.as_ref().map(|shape| Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: format!(
                                "`{}`: {}",
                                shape.render(),
                                shape.dtype.as_deref().unwrap_or("")
                            ),
                        }),
                        range: None,
                    })
                })
            });
            let resp = Response::new_ok(id, result);
            connection.sender.send(Message::Response(resp)).unwrap();
        }
        <DocumentDiagnosticRequest as lsp_types::request::Request>::METHOD => {
            let id = req.id.clone();
            let params: DocumentDiagnosticParams =
                serde_json::from_value(req.params.clone()).unwrap();
            let uri = params.text_document.uri;
            let diagnostics = documents
                .get(&uri)
                .map(|text| {
                    uri.to_file_path()
                        .ok()
                        .map(|p| {
                            analyze_source_at_path_with_cache(text, &p, module_cache).diagnostics
                        })
                        .unwrap_or_else(|| {
                            analyze_source_with_cache(text, module_cache).diagnostics
                        })
                })
                .unwrap_or_default();
            let full = FullDocumentDiagnosticReport {
                result_id: None,
                items: diagnostics,
            };
            let report = DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
                related_documents: None,
                full_document_diagnostic_report: full,
            });
            let resp = Response::new_ok(id, report);
            connection.sender.send(Message::Response(resp)).unwrap();
        }
        _ => {
            let resp = Response::new_err(
                req.id.clone(),
                lsp_server::ErrorCode::MethodNotFound as i32,
                "Unsupported method".into(),
            );
            connection.sender.send(Message::Response(resp)).unwrap();
        }
    }
}

fn publish_diagnostics(
    connection: &Connection,
    documents: &HashMap<Url, String>,
    uri: &Url,
    module_cache: &mut ModuleCache,
) {
    let diagnostics = documents
        .get(uri)
        .map(|text| {
            uri.to_file_path()
                .ok()
                .map(|path| {
                    analyze_source_at_path_with_cache(text, &path, module_cache).diagnostics
                })
                .unwrap_or_else(|| analyze_source_with_cache(text, module_cache).diagnostics)
        })
        .unwrap_or_default();
    let params = PublishDiagnosticsParams {
        uri: uri.clone(),
        diagnostics,
        version: None,
    };
    let notif = lsp_server::Notification::new(
        <PublishDiagnostics as lsp_types::notification::Notification>::METHOD.to_string(),
        params,
    );
    let _ = connection.sender.send(Message::Notification(notif));
}
