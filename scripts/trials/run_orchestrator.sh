#!/usr/bin/env bash
# Canonical task-orchestrator entrypoint. The old run_trial_driver.sh remains as
# a compatibility wrapper until every caller has migrated and been verified.
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec bash "$SCRIPT_DIR/run_trial_driver.sh" "$@"
