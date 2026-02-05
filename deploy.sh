#!/usr/bin/env bash
#
# Hybrid Trader Deployment Script
#
# Usage:
#   ./deploy.sh              # Full deployment
#   ./deploy.sh --no-restart # Deploy without service restart
#   ./deploy.sh --test-only  # Only run tests
#
# This script:
#   1. Pulls latest code from git
#   2. Updates Python dependencies
#   3. Runs the test suite
#   4. Restarts the systemd service (if enabled)
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
VENV_DIR="${PROJECT_DIR}/venv"
SERVICE_NAME="hybrid_trader"
PYTHON_VERSION="python3.13"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        exit 1
    fi
}

# =============================================================================
# Deployment Steps
# =============================================================================

step_git_pull() {
    log_info "Pulling latest changes from git..."

    cd "${PROJECT_DIR}"

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_warning "Uncommitted changes detected. Stashing..."
        git stash
    fi

    # Pull latest
    git pull origin main || git pull origin master || {
        log_error "Failed to pull from git"
        exit 1
    }

    log_success "Git pull complete"
}

step_setup_venv() {
    log_info "Setting up Python virtual environment..."

    cd "${PROJECT_DIR}"

    # Create venv if it doesn't exist
    if [[ ! -d "${VENV_DIR}" ]]; then
        log_info "Creating new virtual environment..."
        ${PYTHON_VERSION} -m venv "${VENV_DIR}" || python3 -m venv "${VENV_DIR}"
    fi

    # Activate venv
    source "${VENV_DIR}/bin/activate"

    # Upgrade pip
    pip install --upgrade pip --quiet

    # Install/update dependencies
    log_info "Installing dependencies..."
    pip install -r requirements.txt --quiet

    log_success "Virtual environment ready"
}

step_run_tests() {
    log_info "Running test suite..."

    cd "${PROJECT_DIR}"
    source "${VENV_DIR}/bin/activate"

    # Run pytest
    if pytest tests/ -v --tb=short; then
        log_success "All tests passed"
    else
        log_error "Tests failed! Aborting deployment."
        exit 1
    fi
}

step_type_check() {
    log_info "Running type checks..."

    cd "${PROJECT_DIR}"
    source "${VENV_DIR}/bin/activate"

    # Run mypy (non-blocking)
    if mypy src/ --ignore-missing-imports --no-error-summary 2>/dev/null; then
        log_success "Type checks passed"
    else
        log_warning "Type check warnings (non-blocking)"
    fi
}

step_restart_service() {
    log_info "Restarting systemd service..."

    # Check if service exists
    if ! systemctl list-unit-files | grep -q "${SERVICE_NAME}.service"; then
        log_warning "Service ${SERVICE_NAME} not found. Skipping restart."
        log_info "To install the service, run:"
        log_info "  sudo cp systemd/${SERVICE_NAME}.service /etc/systemd/system/"
        log_info "  sudo systemctl daemon-reload"
        log_info "  sudo systemctl enable ${SERVICE_NAME}"
        return
    fi

    # Restart the service
    if sudo systemctl restart "${SERVICE_NAME}"; then
        log_success "Service restarted"

        # Wait a moment and check status
        sleep 2
        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            log_success "Service is running"
        else
            log_error "Service failed to start"
            sudo journalctl -u "${SERVICE_NAME}" -n 20 --no-pager
            exit 1
        fi
    else
        log_error "Failed to restart service"
        exit 1
    fi
}

step_show_status() {
    log_info "Deployment Summary"
    echo ""
    echo "  Project:    ${PROJECT_DIR}"
    echo "  Python:     $(${VENV_DIR}/bin/python --version)"
    echo "  Git Branch: $(git branch --show-current)"
    echo "  Git Commit: $(git rev-parse --short HEAD)"
    echo ""

    if systemctl list-unit-files | grep -q "${SERVICE_NAME}.service"; then
        echo "  Service Status:"
        systemctl status "${SERVICE_NAME}" --no-pager -l 2>/dev/null | head -5 || true
    fi

    echo ""
    log_success "Deployment complete!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local no_restart=false
    local test_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-restart)
                no_restart=true
                shift
                ;;
            --test-only)
                test_only=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --no-restart  Deploy without restarting the service"
                echo "  --test-only   Only run tests, no deployment"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo ""
    echo "=========================================="
    echo "   HYBRID TRADER DEPLOYMENT"
    echo "=========================================="
    echo ""

    # Check prerequisites
    check_command git
    check_command pip

    if [[ "${test_only}" == "true" ]]; then
        step_setup_venv
        step_run_tests
        step_type_check
        log_success "Test run complete"
        exit 0
    fi

    # Full deployment
    step_git_pull
    step_setup_venv
    step_run_tests
    step_type_check

    if [[ "${no_restart}" == "false" ]]; then
        step_restart_service
    else
        log_info "Skipping service restart (--no-restart)"
    fi

    step_show_status
}

main "$@"
