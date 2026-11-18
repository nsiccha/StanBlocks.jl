#!/usr/bin/env bash
# StanBlocks app (PosteriorDBWeb, web/) gordito one-off service launcher: runs
# the app in the foreground on loopback port 8091 (default; the first argument
# overrides it for a scratch run). The KB starts it as a one-off service
# (`POST /remote_service_start`, Restart, Switch to latest; skill
# expose-publicly §2b-0) and serves it at /p/StanBlocks/. It runs under
# non-interactive sshd, so PATH is set here, not inherited from an agent shell.
#
# web/app/main.jl (the desktop entry point) binds 0.0.0.0; this launcher calls
# the serve function itself so the service listens on loopback only. No
# Revise: a deployed app loads landed code in a fresh process, and
# --startup-file=no keeps a host startup.jl from loading it.
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.julia/juliaup/bin:/usr/local/bin:/usr/bin:/bin"
# Snippets the /sandbox editor saves (Run, Rename, Delete, Toggle expect) go to
# a folder outside the checkout: the KB restarts and switches the service only
# from a clean checkout, and recreates a lost one. See web/README.md.
export STANBLOCKS_SANDBOX_DIR="${STANBLOCKS_SANDBOX_DIR:-$HOME/.local/share/stanblocks/sandbox}"
mkdir -p "$STANBLOCKS_SANDBOX_DIR"
port="${1:-8091}"
cd "$(dirname "$0")/../.."
# The KB instantiates this committed, pinned environment on every start,
# restart and switch.
# kb-resolve: web/app
exec julia -t 4,1 --startup-file=no --project=web/app -e '
    using PosteriorDBWeb
    PosteriorDBWeb.serve(; host="127.0.0.1", port=parse(Int, ARGS[1]), revise=nothing)
' -- "$port"
