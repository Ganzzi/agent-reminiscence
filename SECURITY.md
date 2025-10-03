# Security Policy

## Supported Versions

Currently supported versions of Agent Mem:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Agent Mem, please report it by emailing the maintainers or creating a private security advisory on GitHub.

**Please do NOT create a public GitHub issue for security vulnerabilities.**

### What to Include

When reporting a security issue, please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Release**: Depends on severity (critical: ASAP, high: 2 weeks, medium: 1 month)

## Security Best Practices

When using Agent Mem:

1. **Never commit `.env` files** with credentials
2. **Use strong passwords** for PostgreSQL and Neo4j
3. **Enable SSL/TLS** for database connections in production
4. **Restrict network access** with firewalls
5. **Keep dependencies updated** regularly
6. **Use secrets management** (AWS Secrets Manager, Vault, etc.)
7. **Monitor logs** for suspicious activity
8. **Regular security audits** of your deployment

## Security Features

Agent Mem includes:

- Environment-based configuration (no hardcoded secrets)
- Connection pooling with proper cleanup
- Input validation with Pydantic models
- Async/await for proper resource management

## Responsible Disclosure

We appreciate responsible disclosure of security issues and will acknowledge security researchers who help improve Agent Mem's security.
