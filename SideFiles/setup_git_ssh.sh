#!/usr/bin/env bash
set -euo pipefail

# --- settings
KEY="${HOME}/.ssh/id_ed25519"
SSH_CONFIG="${HOME}/.ssh/config"

echo "🔧 GitHub SSH setup on macOS"

# 1) Create a key if missing
if [[ ! -f "${KEY}" ]]; then
  echo "🔑 No SSH key found at ${KEY}"
  read -rp "Enter the email to embed in the key (e.g. your GitHub email): " EMAIL
  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"
  ssh-keygen -t ed25519 -C "${EMAIL}" -f "${KEY}"
  echo "✅ Generated ${KEY} and ${KEY}.pub"
else
  echo "✅ Found existing key: ${KEY}"
fi

# 2) Start/attach to ssh-agent (macOS usually runs it already)
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  echo "▶️  Starting/attaching to ssh-agent"
  eval "$(ssh-agent -s)"
fi

# 3) Make macOS remember & auto-load the key
mkdir -p "$(dirname "${SSH_CONFIG}")"
touch "${SSH_CONFIG}"
chmod 600 "${SSH_CONFIG}"

# Append config block if missing
if ! grep -q "AddKeysToAgent yes" "${SSH_CONFIG}" 2>/dev/null; then
  cat >> "${SSH_CONFIG}" <<'EOF'

# --- GitHub / general SSH convenience ---
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
EOF
  echo "📝 Wrote auto-load settings to ${SSH_CONFIG}"
else
  echo "✅ ~/.ssh/config already has AddKeysToAgent/UseKeychain block"
fi

# 4) Add key to the agent and store passphrase in Keychain
if ssh-add -l 2>/dev/null | grep -q "$(ssh-keygen -lf "${KEY}" | awk '{print $2}')"; then
  echo "✅ Key already loaded in agent"
else
  # On macOS this flag stores passphrase in Keychain permanently
  if ssh-add --apple-use-keychain "${KEY}" 2>/dev/null; then
    echo "✅ Key loaded and passphrase stored in Apple Keychain"
  else
    # Fallback if older ssh-add without that flag
    ssh-add "${KEY}"
    echo "ℹ️  Loaded key into agent (no --apple-use-keychain support found)"
  fi
fi

# 5) Put public key on clipboard for GitHub
if command -v pbcopy >/dev/null; then
  pbcopy < "${KEY}.pub"
  echo "📋 Your public key has been copied to the clipboard."
else
  echo "🔎 Your public key is at: ${KEY}.pub"
fi
echo "👉 Add it at: https://github.com/settings/keys (New SSH key)"

# 6) Convert this repo’s remote to SSH if it’s HTTPS
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  current_remote=$(git remote get-url origin 2>/dev/null || true)
  if [[ "${current_remote}" =~ ^https://github.com/.+/.+\.git$ ]]; then
    ssh_remote=$(echo "${current_remote}" | sed -E 's#https://github.com/#git@github.com:#')
    git remote set-url origin "${ssh_remote}"
    echo "🔁 Updated remote:"
    git remote -v
  else
    echo "ℹ️  Remote already SSH or no 'origin' set:"
    git remote -v || true
  fi
else
  echo "ℹ️  Not in a Git repo; skipping remote update."
fi

# 7) Test GitHub auth
echo "🧪 Testing SSH connection to GitHub..."
ssh -T git@github.com || true

echo "🎉 All set. Future Git use should not require re-adding the key."

