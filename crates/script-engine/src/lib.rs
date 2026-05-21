//! Embedded CPython runtime for sadda. Runs Python scripts inside the host
//! Rust process, capturing stdout (and stderr) so callers can display the
//! output in a script panel or surface it programmatically.
//!
//! Phase 0 scope: validate that libpython can be embedded into a Rust binary,
//! run a script, and capture its output. UI integration (the egui script
//! panel widget) and exposing the `sadda` engine API to embedded scripts are
//! follow-ups.
#![warn(missing_docs)]

use pyo3::Python;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};

/// Captured streams from a single [`run_script`] invocation.
#[derive(Debug)]
pub struct ScriptOutput {
    /// Everything the script printed to `sys.stdout`.
    pub stdout: String,
    /// Everything the script printed to `sys.stderr`.
    pub stderr: String,
}

/// Runs `code` in a fresh Python globals namespace with stdout and stderr
/// redirected to in-memory buffers, returning the captured output.
///
/// Each call uses a new globals dict; state does not persist between calls.
/// Acquiring the GIL implicitly initialises the interpreter on first call
/// (thanks to pyo3's `auto-initialize` feature).
pub fn run_script(code: &str) -> PyResult<ScriptOutput> {
    Python::attach(|py| {
        let globals = PyDict::new(py);

        let setup = c_str!(
            "import sys, io\n\
             _sadda_stdout = io.StringIO()\n\
             _sadda_stderr = io.StringIO()\n\
             sys.stdout = _sadda_stdout\n\
             sys.stderr = _sadda_stderr\n"
        );
        py.run(setup, Some(&globals), None)?;

        let code_c = std::ffi::CString::new(code).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "script contained interior NUL byte: {e}"
            ))
        })?;
        py.run(code_c.as_c_str(), Some(&globals), None)?;

        let stdout: String = globals
            .get_item("_sadda_stdout")?
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("stdout buffer missing"))?
            .getattr("getvalue")?
            .call0()?
            .extract()?;
        let stderr: String = globals
            .get_item("_sadda_stderr")?
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("stderr buffer missing"))?
            .getattr("getvalue")?
            .call0()?
            .extract()?;

        Ok(ScriptOutput { stdout, stderr })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_print_output() {
        let out = run_script("print('hello from embedded Python')").unwrap();
        assert_eq!(out.stdout, "hello from embedded Python\n");
        assert_eq!(out.stderr, "");
    }

    #[test]
    fn captures_stderr() {
        let out = run_script("import sys; print('warn', file=sys.stderr)").unwrap();
        assert_eq!(out.stdout, "");
        assert_eq!(out.stderr, "warn\n");
    }

    #[test]
    fn arithmetic_and_print() {
        let out = run_script("print(2 + 3 * 4)").unwrap();
        assert_eq!(out.stdout, "14\n");
    }

    #[test]
    fn syntax_error_surfaces_as_pyerr() {
        let err = run_script("def 123():").unwrap_err();
        let is_syntax_error =
            Python::attach(|py| err.is_instance_of::<pyo3::exceptions::PySyntaxError>(py));
        assert!(is_syntax_error, "got error: {err}");
    }

    #[test]
    fn multiline_script_runs() {
        let code = "\
            xs = list(range(5))\n\
            total = sum(xs)\n\
            print(f'sum 0..4 = {total}')\n\
        ";
        let out = run_script(code).unwrap();
        assert_eq!(out.stdout, "sum 0..4 = 10\n");
    }
}
