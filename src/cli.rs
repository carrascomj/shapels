use crate::analyze_source_at_path;
use lsp_types::DiagnosticSeverity;
use shapels::Analysis;
use std::io::{self, BufWriter, Write};
use std::{env, path::PathBuf};
use std::{fs::read_to_string, path::Path, process::exit};

const HELP: &str = "Language server for torch shapes.

Project home page: https://github.com/carrascomj/shapels

\x1b[4mUsage\x1b[24m: shapels [OPTIONS]
    If no args are provided, shapels stats a language server taking jsonl as
    messages from stdin and outputting to stdout.

\x1b[4mOptions\x1b[24m:
    \x1b[1m-p, --path\x1b[22m <path>   Outputs diagnostics to stdin. If any, returns an exit code of -1.
    \x1b[1m-H, --hover\x1b[22m <path>  Outputs hover information to stdin.
    \x1b[1m-h\x1b[22m                  Prints this help message
";

pub struct CliArgs {
    /// Path to report diagnostics on.
    pub path: Option<PathBuf>,
    /// Path to be hovered over.
    pub hover: Option<PathBuf>,
}

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

pub fn pretty_print_analysis(
    analysis: &Analysis,
    path: Option<&Path>,
    hover: Option<&Path>,
) -> io::Result<()> {
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
            )?;
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
            writeln!(out, "[HOVER] {} {}", location, rendered.trim())?;
        }
    }

    out.flush()
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

fn exit_code(analysis: &Analysis) -> i32 {
    if analysis.diagnostics.is_empty() {
        0
    } else {
        -1
    }
}

fn print_or_exit(result: io::Result<()>) {
    if let Err(err) = result {
        eprintln!("Failed to write output: {err}");
        exit(1);
    }
}

pub fn run_analysis_if_args(cli_args: CliArgs) {
    let diag_path = cli_args.path.as_deref();
    let hover_path = cli_args.hover.as_deref();
    if diag_path.is_none() && hover_path.is_none() {
        return;
    }

    match (diag_path, hover_path) {
        (Some(path), Some(hover)) if path == hover => {
            let analysis = analyze_or_exit(path);
            print_or_exit(pretty_print_analysis(&analysis, Some(path), Some(path)));
            exit(exit_code(&analysis));
        }
        (Some(path), Some(hover)) => {
            let analysis = analyze_or_exit(path);
            print_or_exit(pretty_print_analysis(&analysis, Some(path), None));
            let hover_analysis = analyze_or_exit(hover);
            print_or_exit(pretty_print_analysis(&hover_analysis, None, Some(hover)));
            exit(exit_code(&analysis));
        }
        (Some(path), None) => {
            let analysis = analyze_or_exit(path);
            print_or_exit(pretty_print_analysis(&analysis, Some(path), None));
            exit(exit_code(&analysis));
        }
        (None, Some(hover)) => {
            let analysis = analyze_or_exit(hover);
            print_or_exit(pretty_print_analysis(&analysis, None, Some(hover)));
            exit(0);
        }
        (None, None) => unreachable!(),
    };
}
