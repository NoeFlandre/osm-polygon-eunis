eunis_load_hf_token() {
  local token_file="$1"
  if [[ -z "${HF_TOKEN:-}" && -s "$token_file" ]]; then
    HF_TOKEN="$(<"$token_file")"
    export HF_TOKEN
  fi
}
