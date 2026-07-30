#!/usr/bin/env bash
set -Eeuo pipefail

uv sync --group benchmark

prompt() {
  local message="$1"
  local default_value="$2"
  local result
  read -r -p "${message} [${default_value}]: " result
  printf '%s' "${result:-$default_value}"
}

choose_endpoint() {
  local selection
  echo "选择接口：" >&2
  echo "  1) /v1/completions" >&2
  echo "  2) /v1/chat/completions" >&2
  read -r -p "请输入序号 [1]: " selection
  case "${selection:-1}" in
    1) printf '%s' "/v1/completions" ;;
    2) printf '%s' "/v1/chat/completions" ;;
    *)
      echo "无效选项：${selection}" >&2
      exit 2
      ;;
  esac
}

choose_profile() {
  local selection
  echo "选择压测模式：" >&2
  echo "  1) concurrent：持续保持指定并发数" >&2
  echo "  2) constant：按固定请求/秒发送" >&2
  echo "  3) poisson：按泊松分布发送指定平均请求/秒" >&2
  read -r -p "请输入序号 [1]: " selection
  case "${selection:-1}" in
    1) printf '%s' "concurrent" ;;
    2) printf '%s' "constant" ;;
    3) printf '%s' "poisson" ;;
    *)
      echo "无效选项：${selection}" >&2
      exit 2
      ;;
  esac
}

json_escape() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '%s' "$value"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} 必须是正整数，当前值：${value}" >&2
    exit 2
  fi
}

require_nonnegative_number() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "${name} 必须是非负数字，当前值：${value}" >&2
    exit 2
  fi
}

if ! command -v guidellm >/dev/null 2>&1; then
  echo "未找到 guidellm。请先安装：pip install 'guidellm[recommended]'" >&2
  exit 127
fi

echo "=== GuideLLM 交互式压测 ==="

target=$(prompt "服务地址（不要包含接口路径）" "http://10.199.2.48:3000")
model=$(prompt "请求使用的模型名" "Qwen3.6-27B")
endpoint=$(choose_endpoint)
profile_kind=$(choose_profile)
duration=$(prompt "压测持续时间（秒）" "300")
prompt_tokens=$(prompt "每个请求的输入 token 数" "256")
output_tokens=$(prompt "每个请求的输出 token 数" "128")
warmup=$(prompt "预热比例，0 表示关闭，0.1 表示 10%" "0.1")
cooldown=$(prompt "冷却比例，0 表示关闭，0.1 表示 10%" "0.1")
max_errors=$(prompt "累计多少个错误后停止" "100")
tokenizer_model=$(prompt "Hugging Face tokenizer 名称或本地目录" "Qwen/Qwen3.6-27B")

case "$profile_kind" in
  concurrent)
    load_value=$(prompt "并发请求流数量" "300")
    require_positive_integer "并发数" "$load_value"
    profile_config="kind=concurrent,streams=${load_value},warmup=${warmup},cooldown=${cooldown}"
    ;;
  constant|poisson)
    load_value=$(prompt "平均每秒发送的请求数（RPS）" "10")
    require_nonnegative_number "RPS" "$load_value"
    profile_config="kind=${profile_kind},rate=${load_value},warmup=${warmup},cooldown=${cooldown}"
    ;;
esac

require_positive_integer "持续时间" "$duration"
require_positive_integer "输入 token 数" "$prompt_tokens"
require_positive_integer "输出 token 数" "$output_tokens"
require_positive_integer "错误上限" "$max_errors"
require_nonnegative_number "warmup" "$warmup"
require_nonnegative_number "cooldown" "$cooldown"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  read -r -p "检测到 OPENAI_API_KEY，是否使用？[Y/n]: " use_existing_key
else
  use_existing_key="n"
fi

if [[ "${use_existing_key:-Y}" =~ ^[Nn]$ ]]; then
  read -r -s -p "请输入 API Key（输入内容不会显示）: " api_key
  echo
else
  api_key="$OPENAI_API_KEY"
fi

if [[ -z "$api_key" ]]; then
  echo "API Key 不能为空。" >&2
  exit 2
fi

target_json=$(json_escape "${target%/}")
model_json=$(json_escape "$model")
endpoint_json=$(json_escape "$endpoint")
api_key_json=$(json_escape "$api_key")
backend="{\"kind\":\"openai_http\",\"target\":\"${target_json}\",\"request_format\":\"${endpoint_json}\",\"model\":\"${model_json}\",\"api_key\":\"${api_key_json}\"}"

command_args=(
  uv run
  guidellm run
  --backend "$backend"
  --tokenizer "kind=huggingface_auto,model=${tokenizer_model}"
  --profile "$profile_config"
  --constraint "kind=max_duration,seconds=${duration}"
  --constraint "kind=max_errors,count=${max_errors}"
  --data "kind=synthetic_text,prompt_tokens=${prompt_tokens},output_tokens=${output_tokens}"
)

echo
echo "=== 即将开始 ==="
echo "服务：${target%/}${endpoint}"
echo "模型：${model}"
echo "模式：${profile_config}"
echo "持续：${duration} 秒"
echo "Token：输入 ${prompt_tokens}，输出 ${output_tokens}"
echo "Tokenizer：${tokenizer_model}"
echo
read -r -p "确认执行？[Y/n]: " confirmation

if [[ "${confirmation:-Y}" =~ ^[Nn]$ ]]; then
  echo "已取消。"
  exit 0
fi

exec "${command_args[@]}"
