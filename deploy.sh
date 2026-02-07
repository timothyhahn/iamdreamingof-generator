#!/usr/bin/env bash
set -euo pipefail

readonly APP="iamdreamingof-generator"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FLY_TOML="$SCRIPT_DIR/fly.toml"

info()  { printf '\033[0;32m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
die()   { printf '\033[0;31m==> %s\033[0m\n' "$*" >&2; exit 1; }

# ── Parse flags ──────────────────────────────────────────────────────

SKIP_CONFIRM=false
FLY_DEPLOY_FLAGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)        SKIP_CONFIRM=true ;;
    --remote-only)   FLY_DEPLOY_FLAGS+=(--remote-only) ;;
    *)               die "Unknown flag: $1" ;;
  esac
  shift
done

# ── Pre-flight checks ───────────────────────────────────────────────

[[ -f "$FLY_TOML" ]] || die "fly.toml not found at $FLY_TOML"
command -v jq  >/dev/null || die "'jq' required (brew install jq)"

# fly (homebrew) or flyctl (GitHub Actions / manual install)
if command -v fly >/dev/null 2>&1; then
  FLY=fly
elif command -v flyctl >/dev/null 2>&1; then
  FLY=flyctl
else
  die "'fly' or 'flyctl' CLI not found"
fi

toml_app=$(grep '^app ' "$FLY_TOML" | head -1 | sed "s/^app *= *['\"]//;s/['\"].*//")
[[ "$toml_app" == "$APP" ]] || die "fly.toml app='$toml_app' doesn't match '$APP'"

# ── Step 1: Build & push image (without creating machines) ───────────

IMAGE_LABEL="deploy-$(date -u +%Y%m%d-%H%M%S)"
IMAGE="registry.fly.io/$APP:$IMAGE_LABEL"

info "Building and pushing image ($IMAGE_LABEL)..."

if ! $FLY deploy --build-only --push --image-label "$IMAGE_LABEL" \
    --app "$APP" "${FLY_DEPLOY_FLAGS[@]+"${FLY_DEPLOY_FLAGS[@]}"}"; then
  die "fly deploy failed (see output above)"
fi

info "Image: $IMAGE"

# ── Step 2: Clean up existing machines ───────────────────────────────

info "Listing machines for $APP..."
machines_json=$($FLY machines list --app "$APP" --json 2>/dev/null || echo "[]")
machine_ids=()
while IFS= read -r id; do
  [[ -n "$id" ]] && machine_ids+=("$id")
done < <(echo "$machines_json" | jq -r '.[].id // empty')

if [[ ${#machine_ids[@]} -gt 0 ]]; then
  warn "Found ${#machine_ids[@]} machine(s):"
  echo "$machines_json" | jq -r '.[] | "  \(.id)  \(.name // "-")  state=\(.state)"'

  if [[ "$SKIP_CONFIRM" == "false" ]]; then
    echo
    read -rp "Destroy these machine(s)? [y/N] " confirm
    [[ "$confirm" == [yY] ]] || die "Aborted"
  fi

  for id in "${machine_ids[@]}"; do
    state=$(echo "$machines_json" | jq -r ".[] | select(.id == \"$id\") | .state")
    if [[ "$state" == "started" || "$state" == "running" ]]; then
      info "Stopping machine $id (state=$state)..."
      $FLY machines stop "$id" --app "$APP" --wait-timeout 30s || warn "Could not stop $id, skipping"
    fi
    info "Destroying machine $id..."
    $FLY machines destroy "$id" --app "$APP" || warn "Could not destroy $id — may need manual cleanup"
  done
else
  info "No existing machines to clean up."
fi

# ── Step 3: Create scheduled machine ─────────────────────────────────

# Parse [env] section from fly.toml (handles both single and double quotes)
env_args=()
in_env=false
while IFS= read -r line; do
  [[ "$line" =~ ^\[env\] ]] && { in_env=true; continue; }
  [[ "$line" =~ ^\[       ]] && { in_env=false; continue; }
  if $in_env && [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*= ]]; then
    key="${BASH_REMATCH[1]}"
    value=$(echo "$line" | sed "s/^[^=]*=[[:space:]]*//;s/^['\"]//;s/['\"].*//")
    [[ -n "$value" ]] && env_args+=(--env "$key=$value")
  fi
done < "$FLY_TOML"

# Parse region (handles both quote styles)
region=$(grep '^primary_region' "$FLY_TOML" | head -1 | sed "s/.*=[[:space:]]*['\"]//;s/['\"].*//" || true)

# Parse VM config from [[vm]] section (handles both quote styles)
vm_section=$(sed -n '/^\[\[vm\]\]/,/^\[/p' "$FLY_TOML" || true)
vm_memory=$(echo "$vm_section"   | grep 'memory'   | head -1 | sed "s/.*=[[:space:]]*['\"]//;s/['\"].*//;s/[^0-9]//g" || true)
vm_cpu_kind=$(echo "$vm_section" | grep 'cpu_kind'  | head -1 | sed "s/.*=[[:space:]]*['\"]//;s/['\"].*//" || true)
vm_cpus=$(echo "$vm_section"     | grep 'cpus'      | head -1 | sed "s/.*=[[:space:]]*//;s/[^0-9]//g" || true)

run_args=("$IMAGE" --app "$APP" --schedule daily)
[[ ${#env_args[@]} -gt 0 ]]  && run_args+=("${env_args[@]}")
[[ -n "${region:-}" ]]       && run_args+=(--region "$region")
[[ -n "${vm_memory:-}" ]]    && run_args+=(--vm-memory "$vm_memory")
[[ -n "${vm_cpu_kind:-}" ]]  && run_args+=(--vm-cpu-kind "$vm_cpu_kind")
[[ -n "${vm_cpus:-}" ]]      && run_args+=(--vm-cpus "$vm_cpus")

info "Creating daily scheduled machine..."
info "  $FLY machine run ${run_args[*]}"

# Retry a few times — the registry sometimes needs a moment to propagate the manifest
for attempt in 1 2 3; do
  if $FLY machine run "${run_args[@]}" 2>&1; then
    break
  fi
  if [[ "$attempt" -eq 3 ]]; then
    die "Failed to create scheduled machine after $attempt attempts"
  fi
  warn "Attempt $attempt failed, retrying in 10s (registry propagation delay)..."
  sleep 10
done

# ── Verify ────────────────────────────────────────────────────────────

info "Verifying..."
final_machines=$($FLY machines list --app "$APP" --json 2>/dev/null || echo "[]")
count=$(echo "$final_machines" | jq 'length')
if [[ "$count" -ne 1 ]]; then
  warn "Expected 1 machine, found $count:"
  echo "$final_machines" | jq -r '.[] | "  \(.id)  state=\(.state)"'
  die "Unexpected machine count — check https://fly.io/apps/$APP/machines"
fi

schedule=$(echo "$final_machines" | jq -r '.[0].config.schedule // "none"')
if [[ "$schedule" != "daily" ]]; then
  die "Machine exists but schedule='$schedule', expected 'daily'"
fi

echo "$final_machines" | jq -r '.[] | "  \(.id)  schedule=\(.config.schedule // "none")  image=\(.config.image // "unknown")"'

info "Done! https://fly.io/apps/$APP/monitoring"
