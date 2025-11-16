# Reporting and Fixing Security Issues

Please do not open GitHub issues or pull requests - this makes the problem
immediately visible to everyone, including malicious actors. Security issues in
this open source project can be safely reported contacting support@dreadnode.io.

## Static Analysis with Semgrep

This project uses Semgrep for automated security analysis. The checker runs on:

- All pull requests targeting main
- Direct pushes to main
- Daily scheduled scans

### Enabled Rulesets

- `p/python`: Python-specific checks
- `p/security-audit`: General security best practices
- `p/secrets`: Detection of hardcoded secrets
- `p/owasp-top-ten`: OWASP Top 10 vulnerability checks
- `p/supply-chain`: Supply chain security checks

### Severity Levels

- **ERROR**: Blocking issues that must be fixed
- **WARNING**: Non-blocking issues that should be reviewed
- **INFO**: Informational findings

### Local Usage

Run Semgrep locally before pushing:

```bash
pip install semgrep
semgrep --config=auto
```
