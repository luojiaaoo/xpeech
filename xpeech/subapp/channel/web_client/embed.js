/* Xpeech embedded chat module: no signing keys, cookies, or persistent tokens. */
var origin = new URL(import.meta.url).origin;
var endpoint = origin + "/api/auth/embed/session";
var hostOrigin = window.location.origin;
var host = null;
var root = null;
var panel = null;
var launcher = null;
var closeButton = null;
var body = null;
var status = null;
var frame = null;
var frameReady = false;
var currentId = null;
var channelName = null;
var userIdWasSet = false;
var session = null;
var version = 0;
var controller = null;
var refreshTimer = null;
var frameTimer = null;
var needsRetry = false;

function initialize() {
  if (host) return;
  host = document.createElement("div");
  host.id = "xpeech-widget";
  root = host.attachShadow({ mode: "open" });
  root.innerHTML = '<style>' +
  ':host{all:initial;position:fixed;right:20px;bottom:20px;z-index:2147483000;font-family:system-ui,sans-serif;color:#18213d}' +
  '*{box-sizing:border-box}button{font:inherit;cursor:pointer}' +
  '#launcher{width:58px;height:58px;border:0;border-radius:50%;background:#3370ff;color:white;box-shadow:0 6px 24px #152c6740;font-size:24px}' +
  '#panel{display:none;position:absolute;right:0;bottom:76px;width:min(440px,calc(100vw - 40px));height:min(700px,calc(100dvh - 120px));background:white;border:1px solid #dfe5ef;border-radius:16px;box-shadow:0 12px 48px #152c6733;overflow:hidden;flex-direction:column}' +
  '#panel.open{display:flex}header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #edf0f5;flex-shrink:0}' +
  '#close{background:none;border:0;font-size:24px;color:#5b6578}#body{flex:1;min-height:0;position:relative}' +
  '#status{padding:24px;line-height:1.6;font-size:14px}iframe{display:block;width:100%;height:100%;border:0}' +
  'button:focus-visible{outline:3px solid #8ab4ff;outline-offset:3px}[hidden]{display:none!important}' +
  '</style><section id="panel" role="dialog" aria-label="AI 助手"><header><strong>AI 助手</strong><button id="close" aria-label="关闭聊天">×</button></header><div id="body"><div id="status" role="status">正在连接…</div></div></section><button id="launcher" aria-label="打开 AI 助手" aria-expanded="false">💬</button>';
  document.body.appendChild(host);
  panel = root.getElementById("panel");
  launcher = root.getElementById("launcher");
  closeButton = root.getElementById("close");
  body = root.getElementById("body");
  status = root.getElementById("status");
  launcher.addEventListener("click", function () {
    if (panel.classList.contains("open")) close(); else open();
  });
  closeButton.addEventListener("click", close);
  root.addEventListener("keydown", function (event) {
    if (event.key === "Escape") close();
  });
  window.addEventListener("message", onMessage);
}

function open() {
  if (!host) return;
  panel.classList.add("open");
  launcher.setAttribute("aria-expanded", "true");
  ensureConnected();
  closeButton.focus();
}
function close() {
  if (!host) return;
  panel.classList.remove("open");
  launcher.setAttribute("aria-expanded", "false");
  if (!launcher.hidden) launcher.focus();
}
function hideButton() {
  if (launcher) launcher.hidden = true;
}
function showButton() {
  if (launcher) launcher.hidden = false;
}

function showStatus(text) {
  status.textContent = text;
  status.hidden = false;
}
function report(error) {
  needsRetry = true;
  var detail = error instanceof Error ? error.message : String(error);
  showStatus("连接失败：" + detail + "。请点击按钮重试。");
  window.dispatchEvent(new CustomEvent("xpeech:error", { detail: { message: detail } }));
}
function revoke(token) {
  if (!token) return;
  fetch(endpoint, {
    method: "DELETE", credentials: "omit", keepalive: true,
    headers: { Authorization: "Bearer " + token }
  }).catch(function () { /* The server also expires abandoned sessions. */ });
}
function removeFrame() {
  clearTimeout(frameTimer);
  frameReady = false;
  if (frame) frame.remove();
  frame = null;
}
function createFrame() {
  if (frame) return;
  frame = document.createElement("iframe");
  frame.title = "AI 助手聊天";
  var url = new URL("/", origin);
  url.searchParams.set("embed", "1");
  url.searchParams.set("parent_origin", hostOrigin);
  frame.src = url.href;
  frame.referrerPolicy = "no-referrer";
  body.appendChild(frame);
  frameTimer = setTimeout(function () {
    if (!frameReady) {
      removeFrame();
      report(new Error("聊天页面未能加载，请检查前端构建和嵌入来源配置"));
    }
  }, 20000);
}
function deliver() {
  if (!frame || !frameReady || !session) return;
  frame.contentWindow.postMessage({
    type: "xpeech:session", token: session.token, channel_name: channelName
  }, origin);
}
function onMessage(event) {
  if (event.origin !== origin || !frame || event.source !== frame.contentWindow) return;
  var data = event.data;
  if (!data || typeof data !== "object") return;
  if (data.type === "xpeech:ready") {
    frameReady = true;
    clearTimeout(frameTimer);
    deliver();
  } else if (data.type === "xpeech:authenticated") {
    if (session) status.hidden = true;
  } else if (data.type === "xpeech:expired") {
    refresh();
  } else if (data.type === "xpeech:error") {
    report(new Error(typeof data.message === "string" ? data.message : "登录失败"));
  }
}
async function connect(expectedVersion) {
  if (controller || !currentId || !host) return;
  var pending = new AbortController();
  controller = pending;
  var previous = session;
  try {
    var headers = { "Content-Type": "application/json" };
    if (previous) headers.Authorization = "Bearer " + previous.token;
    var response = await fetch(endpoint, {
      method: "POST", credentials: "omit", headers: headers,
      body: JSON.stringify({ user_id: currentId, channel_name: channelName }),
      signal: pending.signal
    });
    var result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "无法建立用户会话");
    if (typeof result.token !== "string" || !result.token.startsWith("embed_") ||
        !Number.isFinite(result.expires_in) || result.expires_in <= 0) {
      throw new Error("无效的会话响应");
    }
    if (expectedVersion !== version || !host) {
      revoke(result.token);
      return;
    }
    session = result;
    needsRetry = false;
    createFrame();
    deliver();
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, Math.max(1000, (result.expires_in - 30) * 1000));
  } catch (error) {
    if (expectedVersion === version && host && error.name !== "AbortError") report(error);
  } finally {
    if (controller === pending) controller = null;
  }
}
function refresh() {
  if (currentId && host) connect(version);
}
function normalize(value) {
  if ((typeof value !== "string" && typeof value !== "number") ||
      (typeof value === "number" && !Number.isFinite(value))) {
    throw new TypeError("setUserId 需要非空字符串或有效数字");
  }
  var id = String(value).trim();
  if (!id || /[\u0000-\u001f]/.test(id)) {
    throw new TypeError("user_id 不能为空或包含控制字符");
  }
  if (Array.from(id).length > 128) throw new RangeError("user_id 最多 128 个字符");
  return id;
}
function ensureConnected() {
  if (host && currentId && (!session || !frame || needsRetry) && !controller) {
    showStatus("正在连接…");
    if (session && !needsRetry) { createFrame(); deliver(); } else connect(version);
  }
}
function setChannelName(value) {
  if (channelName !== null || userIdWasSet) {
    throw new Error("setChannelName 只能设置一次，且必须在首次 setUserId 之前调用");
  }
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/.test(value.trim())) {
    throw new TypeError("channel_name 需要 1–64 个字符，仅支持字母、数字、点、下划线、冒号和连字符");
  }
  if (["internal", "feishu", "web_client"].includes(value.trim())) {
    throw new TypeError("嵌入组件必须使用自定义通道，不能使用内置通道");
  }
  channelName = value.trim();
}
function setUserId(value) {
  var next = normalize(value);
  if (channelName === null) throw new Error("必须先调用 setChannelName 设置嵌入通道");
  userIdWasSet = true;
  if (host && next === currentId) {
    ensureConnected();
    return;
  }
  initialize();
  currentId = next;
  version += 1;
  clearTimeout(refreshTimer);
  if (controller) controller.abort();
  controller = null;
  var old = session;
  session = null;
  removeFrame();
  if (old) revoke(old.token);
  needsRetry = false;
  showButton();
  showStatus("正在连接…");
  connect(version);
}
function destroy() {
  version += 1;
  clearTimeout(refreshTimer);
  if (controller) controller.abort();
  controller = null;
  if (session) revoke(session.token);
  session = null;
  currentId = null;
  needsRetry = false;
  removeFrame();
  window.removeEventListener("message", onMessage);
  if (host) host.remove();
  host = root = panel = launcher = closeButton = body = status = null;
}

export { setChannelName, setUserId, hideButton, showButton, open, close, destroy };
