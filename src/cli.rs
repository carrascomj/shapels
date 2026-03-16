use lsp_types::DiagnosticSeverity;
use shapels::Analysis;
use shapels::analyze_source_at_path;
use std::io::{self, BufWriter, Write};
use std::{env, path::PathBuf};
use std::{fs::read_to_string, path::Path, process::exit};

const HELP: &str = "Language server for torch shapes.

Project home page: https://github.com/carrascomj/shapels

\x1b[4mUsage\x1b[24m: shapels [OPTIONS]
    If no args are provided, shapels starts a language server taking JSONL as
    messages from stdin and outputting to stdout.

\x1b[4mOptions\x1b[24m:
    \x1b[1m-p, --path\x1b[22m <path>   Outputs diagnostics to stdin. If any, returns an exit code of -1.
    \x1b[1m-H, --hover\x1b[22m <path>  Outputs hover information to stdin.
    \x1b[1m-h\x1b[22m                  Prints this help message.
";

pub struct CliArgs {
    /// Path to report diagnostics on.
    pub path: Option<PathBuf>,
    /// Path to be hovered over.
    pub hover: Option<PathBuf>,
}

#[derive(Debug)]
pub enum CliError {
    FoundDiagnostics,
    IoError(io::Error),
}

/// Extract and parse the CLI args into [`CliArgs`].
pub fn parse_args() -> CliArgs {
    let mut args = env::args().skip(1);
    let mut path = None;
    let mut hover = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                exit(0);
            }
            "-p" | "--path" => path = Some(expect_path_value(&mut args, &arg)),
            "-H" | "--hover" => hover = Some(expect_path_value(&mut args, &arg)),
            other => eprintln!("Ignoring unknown argument: {other}"),
        }
    }

    CliArgs { path, hover }
}

fn expect_path_value<I>(args: &mut I, flag: &str) -> PathBuf
where
    I: Iterator<Item = String>,
{
    match args.next() {
        Some(value) => PathBuf::from(value),
        None => {
            eprintln!("Expected a path after `{flag}`");
            exit(1);
        }
    }
}

fn severity_label(severity: Option<DiagnosticSeverity>) -> &'static str {
    match severity {
        Some(DiagnosticSeverity::ERROR) => "ERROR",
        Some(DiagnosticSeverity::WARNING) => "WARN",
        Some(DiagnosticSeverity::INFORMATION) => "INFO",
        Some(DiagnosticSeverity::HINT) => "HINT",
        _ => "UNKN",
    }
}

fn format_location(path: &Path, line: u32, character: u32) -> String {
    format!(
        "{}:{}:{}",
        path.display(),
        line.saturating_add(1),
        character.saturating_add(1)
    )
}

fn pretty_print_analysis(
    analysis: &Analysis,
    path: Option<&Path>,
    hover: Option<&Path>,
) -> Result<(), CliError> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    if let Some(p) = path {
        for diag in &analysis.diagnostics {
            let location = format_location(p, diag.range.start.line, diag.range.start.character);
            writeln!(
                out,
                "[{}] {} {}",
                severity_label(diag.severity),
                location,
                diag.message.trim()
            )
            .map_err(CliError::IoError)?;
        }
    }

    if let Some(p) = hover {
        for (range, hover_info) in &analysis.hover_entries {
            let location = format_location(p, range.start.line, range.start.character);
            let rendered = hover_info
                .shape
                .as_ref()
                .map(|s| s.render())
                .unwrap_or_default();
            writeln!(out, "[HOVER] {} {}", location, rendered.trim()).map_err(CliError::IoError)?;
        }
    }

    out.flush().map_err(CliError::IoError)?;
    if analysis.diagnostics.is_empty() {
        Err(CliError::FoundDiagnostics)
    } else {
        Ok(())
    }
}

fn analyze_or_exit(path: &Path) -> Analysis {
    match read_to_string(path) {
        Ok(source) => analyze_source_at_path(&source, path),
        Err(err) => {
            eprintln!("Failed to read {}: {err}", path.display());
            exit(1);
        }
    }
}

fn with_exit_code(result: Result<(), CliError>) -> i32 {
    match result {
        Err(err) => {
            if let CliError::IoError(msg) = err {
                eprintln!("Failed to write output: {msg}");
            }
            -1
        }
        Ok(_) => 0,
    }
}

/// Analyze one or two files, depending on [`CliArgs`], and print diagnostics
/// and/or hover events to the screen.
///
/// If no arguments were provided, it is a noop. Otherwise, it will analyse the
/// file(s) and exit, with -1 if any diagnostics were emitted and 0 otherwise;
/// always 0 if `cli_args.path` is `None`.
pub fn run_analysis_if_args(cli_args: CliArgs) {
    match (cli_args.path.as_deref(), cli_args.hover.as_deref()) {
        (Some(path), Some(hover)) if path != hover => {
            let analysis = analyze_or_exit(path);
            let exit_code = with_exit_code(pretty_print_analysis(&analysis, Some(path), None));
            let hover_analysis = analyze_or_exit(hover);
            with_exit_code(pretty_print_analysis(&hover_analysis, None, Some(hover)));
            exit(exit_code)
        }
        (a @ Some(path), b @ None) | (a @ None, b @ Some(path)) | (a @ Some(path), b @ Some(_)) => {
            let analysis = analyze_or_exit(path);
            exit(with_exit_code(pretty_print_analysis(&analysis, a, b)));
        }
        (None, None) => return,
    };
}
