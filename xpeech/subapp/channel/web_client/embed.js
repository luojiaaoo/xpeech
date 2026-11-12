/* Xpeech embedded chat module: no signing keys, cookies, or persistent tokens. */
var origin = new URL(import.meta.url).origin;
var endpoint = origin + "/api/auth/embed";
var hostOrigin = window.location.origin;
var host = null;
var root = null;
var panel = null;
var launcher = null;
var closeButton = null;
var minimizeButton = null;
var maximizeButton = null;
var body = null;
var status = null;
var frame = null;
var frameReady = false;
var currentId = null;
var channelName = null;
var currentJWT = null;
var jwtProvider = null;
var currentSenderName;
var currentTitle = "AI 助手";
var currentSessionMetadata = {};
var identityConfigured = false;
var verifiedJWT = null;
var version = 0;
var controller = null;
var refreshTimer = null;
var frameTimer = null;
var needsRetry = false;
var drag = null;
var resize = null;
var panelSize = null;
var maximized = false;
var normalPanelBounds = null;
var suppressLauncherClick = false;
var interactionSession = null;
var requestSequence = 0;
var pendingRequests = new Map();
var replyListeners = new Set();
var chatFileListeners = new Set();

function rejectRequests(message, sentOnly = false, keepRequestId = null) {
  pendingRequests.forEach(function (pending, id) {
    if (id === keepRequestId) return;
    if (sentOnly && !pending.sent) return;
    clearTimeout(pending.timer);
    pendingRequests.delete(id);
    pending.reject(new Error(message));
  });
}
function flushRequests() {
  if (!frame || !interactionSession || !verifiedJWT) return;
  for (var pending of pendingRequests.values()) {
    if (pending.sent) {
      if (pending.data.method === 'newConversation') return;
      continue;
    }
    pending.sent = true;
    pending.sessionId = interactionSession;
    frame.contentWindow.postMessage(Object.assign({}, pending.data, { session_id: interactionSession }), origin);
    if (pending.data.method === 'newConversation') return;
  }
}
function requestInteraction(method, text, options = {}) {
  if (!identityConfigured) return Promise.reject(new Error('请先调用 init()'));
  var fileRequest = method === 'downloadFile' || method === 'getFile';
  if (typeof text !== 'string') {
    return Promise.reject(new TypeError(fileRequest ? 'path 必须为字符串' : 'text 必须为字符串'));
  }
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    return Promise.reject(new TypeError('options 必须为对象'));
  }
  if (fileRequest && !text.trim()) {
    return Promise.reject(new TypeError('文件路径不能为空'));
  }
  var files = [];
  if ((method === 'chatSync' || method === 'addFiles') && options.files !== undefined) {
    if (!Array.isArray(options.files)) return Promise.reject(new TypeError('files 必须为 File 数组'));
    files = Array.from(options.files);
    if (!files.every(function (file) { return file instanceof File; })) {
      return Promise.reject(new TypeError('files 必须为 File 数组'));
    }
  }
  if (method === 'chatSync' && !text.trim() && !files.length) {
    return Promise.reject(new TypeError('聊天消息不能为空'));
  }
  var timeout = options.timeoutMs === undefined ? (method === 'chatSync' ? 300000 : 30000) : options.timeoutMs;
  if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 2147483647) {
    return Promise.reject(new TypeError('timeoutMs 必须为有效的正数'));
  }
  return new Promise(function (resolve, reject) {
    var id = String(++requestSequence);
    var pending = {
      data: { type: 'xpeech:request', request_id: id, method: method, text: text, files: files },
      resolve: resolve, reject: reject, sent: false, sessionId: null,
      timer: setTimeout(function () {
        pendingRequests.delete(id);
        reject(new Error('iframe 交互超时'));
        flushRequests();
      }, timeout)
    };
    pendingRequests.set(id, pending);
    ensureConnected();
    flushRequests();
  });
}
function setInput(text, options) { return requestInteraction('setInput', text, options); }
function addFiles(files, options = {}) {
  if (!Array.isArray(files)) return Promise.reject(new TypeError('files 必须为 File 数组'));
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    return Promise.reject(new TypeError('options 必须为对象'));
  }
  return requestInteraction('addFiles', '', Object.assign({}, options, { files: files }));
}
function newConversation(options) { return requestInteraction('newConversation', '', options); }
function send(options) { return requestInteraction('send', '', options); }
function fileInfo(path) {
  var url = new URL('/api/files', origin);
  url.searchParams.set('path', path);
  return { path: path, url: url.href };
}
function chatSync(text, options) {
  return requestInteraction('chatSync', text, options).then(function (result) {
    var files = result.events
      .filter(function (event) { return event.event === 'send_file' && typeof event.context === 'string' && event.context.trim(); })
      .map(function (event) { return fileInfo(event.context); });
    return Object.assign({}, result, { files: files });
  });
}
function downloadFile(path, options) { return requestInteraction('downloadFile', path, options); }
function getFile(path, options) {
  return requestInteraction('getFile', path, options).then(function (result) { return result.file; });
}
function onReply(listener) {
  if (typeof listener !== 'function') throw new TypeError('listener 必须为函数');
  replyListeners.add(listener);
  return function () { replyListeners.delete(listener); };
}
function onChatFile(listener) {
  if (typeof listener !== 'function') throw new TypeError('listener 必须为函数');
  chatFileListeners.add(listener);
  return function () { chatFileListeners.delete(listener); };
}

function initialize() {
  if (host) return;
  host = document.createElement("div");
  host.id = "xpeech-widget";
  root = host.attachShadow({ mode: "open" });
  root.innerHTML = '<style>' +
  ':host{all:initial;position:fixed;right:20px;bottom:20px;width:46px;height:46px;z-index:2147483000;font-family:system-ui,sans-serif;color:#18213d;pointer-events:none}' +
  '*{box-sizing:border-box}button{font:inherit;cursor:pointer}' +
  '#launcher{display:grid;place-items:center;width:46px;height:46px;padding:0;border:0;border-radius:50%;background:#3370ff;color:white;box-shadow:0 4px 16px #152c6730;pointer-events:auto;touch-action:none;user-select:none;transition:transform .18s,background .18s,box-shadow .18s;animation:xpeech-enter .32s ease-out,xpeech-breathe 3.2s ease-in-out .4s 3}' +
  '#launcher svg{width:22px;height:22px;pointer-events:none}' +
  '@media(hover:hover){#launcher:hover{background:#2862ed;transform:translateY(-2px);box-shadow:0 6px 20px #152c6740}}' +
  '#launcher.positioned{animation:none}' +
  '#launcher.dragging{cursor:grabbing;transform:scale(.96);animation:none;transition:none}' +
  '@keyframes xpeech-enter{from{opacity:0;transform:translateY(8px) scale(.85)}to{opacity:1;transform:translateY(0) scale(1)}}' +
  '@keyframes xpeech-breathe{50%{box-shadow:0 4px 20px #3370ff45,0 0 0 5px #3370ff0a}}' +
  '@media(prefers-reduced-motion:reduce){#launcher{animation:none;transition:none}#launcher:hover{transform:none}}' +
  '#panel{display:none;position:fixed;width:min(440px,calc(100vw - 24px));height:min(700px,calc(100dvh - 100px));background:white;border:1px solid #dfe5ef;border-radius:16px;box-shadow:0 12px 48px #152c6733;overflow:hidden;flex-direction:column;pointer-events:auto}' +
  '#panel.open{display:flex}header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #edf0f5;flex-shrink:0}' +
  'header strong{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.window-actions{display:flex;gap:4px;margin-left:12px}#panel.maximized{border-radius:0}#panel.maximized .resize-handle{display:none}' +
  '#panel.open.maximized~#launcher{visibility:hidden}' +
  '#panel.resizing{user-select:none}#panel.resizing iframe{pointer-events:none}' +
  '.resize-handle{position:absolute;z-index:3;touch-action:none;user-select:none}' +
  '.resize-n,.resize-s{left:14px;right:14px;height:6px;cursor:ns-resize}.resize-n{top:0}.resize-s{bottom:0}' +
  '.resize-e,.resize-w{top:14px;bottom:14px;width:6px;cursor:ew-resize}.resize-e{right:0}.resize-w{left:0}' +
  '.resize-ne,.resize-nw,.resize-se,.resize-sw{width:14px;height:14px}.resize-ne,.resize-sw{cursor:nesw-resize}.resize-nw,.resize-se{cursor:nwse-resize}' +
  '.resize-ne,.resize-nw{top:0}.resize-se,.resize-sw{bottom:0}.resize-ne,.resize-se{right:0}.resize-nw,.resize-sw{left:0}' +
  '.window-actions button{display:grid;place-items:center;width:30px;height:30px;background:none;border:0;border-radius:6px;font-size:22px;color:#5b6578}.window-actions button:hover{background:#f0f3f8}#body{flex:1;min-height:0;position:relative}' +
  '#status{position:absolute;inset:0;z-index:1;display:flex;align-items:center;justify-content:center;padding:24px;background:#fff;color:#68788f;line-height:1.6;font-size:14px;text-align:center;overflow:auto}iframe{display:block;width:100%;height:100%;border:0}' +
  'button:focus-visible{outline:3px solid #8ab4ff;outline-offset:3px}[hidden]{display:none!important}' +
  '</style><section id="panel" role="dialog"><header><strong id="title"></strong><div class="window-actions"><button id="minimize" aria-label="最小化聊天" title="最小化">−</button><button id="maximize" aria-label="最大化聊天" title="最大化">□</button><button id="close" aria-label="关闭聊天" title="关闭">×</button></div></header><div id="body"><div id="status" role="status">正在连接…</div></div></section><button id="launcher" aria-expanded="false" title="点击打开聊天，拖动调整位置"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5 9 9 0 0 1-4-.9L3 21l1.9-5.5a9 9 0 0 1-.9-4A8.5 8.5 0 0 1 12.5 3a8.5 8.5 0 0 1 8.5 8.5Z"/><path d="M8 11.5h.01M12 11.5h.01M16 11.5h.01"/></svg></button>';
  document.body.appendChild(host);
  panel = root.getElementById("panel");
  launcher = root.getElementById("launcher");
  closeButton = root.getElementById("close");
  minimizeButton = root.getElementById("minimize");
  maximizeButton = root.getElementById("maximize");
  body = root.getElementById("body");
  status = root.getElementById("status");
  ["n", "e", "s", "w", "ne", "nw", "se", "sw"].forEach(function (edge) {
    var handle = document.createElement("div");
    handle.className = "resize-handle resize-" + edge;
    handle.setAttribute("aria-hidden", "true");
    handle.addEventListener("pointerdown", function (event) { startResize(event, edge); });
    handle.addEventListener("pointermove", moveResize);
    handle.addEventListener("pointerup", endResize);
    handle.addEventListener("pointercancel", endResize);
    handle.addEventListener("lostpointercapture", endResize);
    panel.appendChild(handle);
  });
  root.getElementById("title").textContent = currentTitle;
  panel.setAttribute("aria-label", currentTitle);
  launcher.setAttribute("aria-label", "打开 " + currentTitle);
  launcher.addEventListener("click", function () {
    if (suppressLauncherClick) {
      suppressLauncherClick = false;
      return;
    }
    if (panel.classList.contains("open")) close(); else open();
  });
  launcher.addEventListener("pointerdown", startDrag);
  launcher.addEventListener("pointermove", moveDrag);
  launcher.addEventListener("pointerup", endDrag);
  launcher.addEventListener("pointercancel", endDrag);
  launcher.addEventListener("lostpointercapture", endDrag);
  closeButton.addEventListener("click", close);
  minimizeButton.addEventListener("click", minimize);
  maximizeButton.addEventListener("click", function () {
    setMaximized(!maximized);
  });
  root.addEventListener("keydown", function (event) {
    if (event.key === "Escape") close();
  });
  window.addEventListener("message", onMessage);
  window.addEventListener("resize", keepInViewport);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), Math.max(min, max));
}
function positionPanel() {
  if (!host || !panel.classList.contains("open")) return;
  if (maximized) {
    panel.style.left = "0px";
    panel.style.top = "0px";
    panel.style.width = window.innerWidth + "px";
    panel.style.height = window.innerHeight + "px";
    return;
  }
  if (panelSize) {
    panel.style.width = Math.min(Math.max(panelSize.width, 320), Math.max(0, window.innerWidth - 24)) + "px";
    panel.style.height = Math.min(Math.max(panelSize.height, 360), Math.max(0, window.innerHeight - 24)) + "px";
  }
  var rect = host.getBoundingClientRect();
  var width = panel.offsetWidth;
  var height = panel.offsetHeight;
  var top = rect.top >= window.innerHeight - rect.bottom ? rect.top - height - 12 : rect.bottom + 12;
  panel.style.left = clamp(rect.right - width, 12, window.innerWidth - width - 12) + "px";
  panel.style.top = clamp(top, 12, window.innerHeight - height - 12) + "px";
}
function moveLauncher(left, top) {
  host.style.right = window.innerWidth - clamp(left, 12, window.innerWidth - 58) - 46 + "px";
  host.style.bottom = window.innerHeight - clamp(top, 12, window.innerHeight - 58) - 46 + "px";
  positionPanel();
}
function keepInViewport() {
  if (!host) return;
  if (resize) endResize({ pointerId: resize.id });
  var rect = host.getBoundingClientRect();
  moveLauncher(rect.left, rect.top);
}
function startDrag(event) {
  if (!event.isPrimary || event.button !== 0 || drag || resize) return;
  var rect = host.getBoundingClientRect();
  suppressLauncherClick = false;
  drag = { id: event.pointerId, x: event.clientX, y: event.clientY, left: rect.left, top: rect.top, moved: false };
  launcher.setPointerCapture(event.pointerId);
}
function moveDrag(event) {
  if (!drag || event.pointerId !== drag.id) return;
  var dx = event.clientX - drag.x;
  var dy = event.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 6) return;
  drag.moved = true;
  launcher.classList.add("positioned");
  launcher.classList.add("dragging");
  moveLauncher(drag.left + dx, drag.top + dy);
}
function endDrag(event) {
  if (!drag || event.pointerId !== drag.id) return;
  suppressLauncherClick = drag.moved && event.type === "pointerup";
  drag = null;
  launcher.classList.remove("dragging");
  if (launcher.hasPointerCapture(event.pointerId)) launcher.releasePointerCapture(event.pointerId);
}
function startResize(event, edge) {
  if (maximized) return;
  if (!event.isPrimary || event.button !== 0 || drag || resize || !panel.classList.contains("open")) return;
  var rect = panel.getBoundingClientRect();
  resize = {
    id: event.pointerId, edge: edge, handle: event.currentTarget,
    x: event.clientX, y: event.clientY,
    left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom
  };
  event.preventDefault();
  panel.classList.add("resizing");
  panel.style.cursor = edge === "n" || edge === "s" ? "ns-resize"
    : edge === "e" || edge === "w" ? "ew-resize"
    : edge === "ne" || edge === "sw" ? "nesw-resize" : "nwse-resize";
  resize.handle.setPointerCapture(event.pointerId);
}
function moveResize(event) {
  if (!resize || event.pointerId !== resize.id) return;
  var dx = event.clientX - resize.x;
  var dy = event.clientY - resize.y;
  var left = resize.left;
  var top = resize.top;
  var right = resize.right;
  var bottom = resize.bottom;
  var minWidth = Math.min(320, window.innerWidth - 24);
  var minHeight = Math.min(360, window.innerHeight - 24);
  if (resize.edge.includes("w")) left = clamp(left + dx, 12, right - minWidth);
  if (resize.edge.includes("e")) right = clamp(right + dx, left + minWidth, window.innerWidth - 12);
  if (resize.edge.includes("n")) top = clamp(top + dy, 12, bottom - minHeight);
  if (resize.edge.includes("s")) bottom = clamp(bottom + dy, top + minHeight, window.innerHeight - 12);
  panelSize = { width: right - left, height: bottom - top };
  panel.style.left = left + "px";
  panel.style.top = top + "px";
  panel.style.width = panelSize.width + "px";
  panel.style.height = panelSize.height + "px";
}
function endResize(event) {
  if (!resize || event.pointerId !== resize.id) return;
  var handle = resize.handle;
  resize = null;
  panel.classList.remove("resizing");
  panel.style.cursor = "";
  if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
}
function open() {
  if (!host) return;
  panel.classList.add("open");
  positionPanel();
  launcher.setAttribute("aria-expanded", "true");
  ensureConnected();
  closeButton.focus();
}
function close() {
  if (!host) return;
  if (resize) endResize({ pointerId: resize.id });
  panel.classList.remove("open");
  launcher.setAttribute("aria-expanded", "false");
  if (!launcher.hidden) launcher.focus();
}
function setMaximized(value) {
  if (!host || maximized === value) return;
  if (resize) endResize({ pointerId: resize.id });
  if (value) normalPanelBounds = panel.getBoundingClientRect();
  maximized = value;
  if (value) panel.classList.add('maximized');
  else panel.classList.remove('maximized');
  maximizeButton.textContent = value ? '❐' : '□';
  maximizeButton.setAttribute('aria-label', value ? '还原聊天窗口' : '最大化聊天');
  maximizeButton.title = value ? '还原' : '最大化';
  if (value) positionPanel();
  else if (normalPanelBounds) {
    var width = Math.min(normalPanelBounds.width, Math.max(0, window.innerWidth - 24));
    var height = Math.min(normalPanelBounds.height, Math.max(0, window.innerHeight - 24));
    panelSize = { width: width, height: height };
    panel.style.width = width + 'px';
    panel.style.height = height + 'px';
    panel.style.left = clamp(normalPanelBounds.left, 12, window.innerWidth - width - 12) + 'px';
    panel.style.top = clamp(normalPanelBounds.top, 12, window.innerHeight - height - 12) + 'px';
  }
}
function maximize() {
  if (!host) return;
  open();
  setMaximized(true);
}
function restore() {
  if (!host) return;
  setMaximized(false);
  if (!panel.classList.contains('open')) open();
}
function minimize() { close(); }
function hideButton() {
  if (launcher) {
    if (drag) endDrag({ pointerId: drag.id, type: "pointercancel" });
    launcher.hidden = true;
  }
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
  rejectRequests(detail);
  showStatus("连接失败：" + detail + "。请点击按钮重试。");
  window.dispatchEvent(new CustomEvent("xpeech:error", { detail: { message: detail } }));
}
function removeFrame() {
  clearTimeout(frameTimer);
  frameReady = false;
  interactionSession = null;
  rejectRequests('聊天窗口已关闭');
  if (frame) frame.remove();
  frame = null;
}
function createFrame() {
  if (frame) return;
  frame = document.createElement("iframe");
  frame.title = currentTitle + "聊天";
  var url = new URL("/", origin);
  url.searchParams.set("embed", "1");
  url.searchParams.set("parent_origin", hostOrigin);
  frame.src = url.href;
  frame.referrerPolicy = "no-referrer";
  body.appendChild(frame);
  frameTimer = setTimeout(function () {
    if (!frameReady) {
      removeFrame();
      report(new Error("聊天页面未能加载，请检查前端构建和服务地址"));
    }
  }, 20000);
}
function deliver() {
  if (!frame || !frameReady || !verifiedJWT) return;
  frame.contentWindow.postMessage({
    type: "xpeech:credentials", jwt: verifiedJWT, user_id: currentId,
    sender_name: currentSenderName, title: currentTitle, session_metadata: currentSessionMetadata
  }, origin);
}
function onMessage(event) {
  if (event.origin !== origin || !frame || event.source !== frame.contentWindow) return;
  var data = event.data;
  if (!data || typeof data !== "object") return;
  if (data.type === "xpeech:ready") {
    interactionSession = null;
    rejectRequests('聊天页面已重新加载', true);
    frameReady = true;
    clearTimeout(frameTimer);
    deliver();
  } else if (data.type === "xpeech:authenticated") {
    if (verifiedJWT) status.hidden = true;
  } else if (data.type === "xpeech:interaction-ready" && typeof data.session_id === 'string') {
    if (interactionSession && interactionSession !== data.session_id) rejectRequests('会话已切换', true);
    interactionSession = data.session_id;
    pendingRequests.forEach(function (pending, id) {
      if (!pending.nextSessionId) return;
      clearTimeout(pending.timer);
      pendingRequests.delete(id);
      if (pending.nextSessionId === interactionSession) pending.resolve({ session_id: interactionSession });
      else pending.reject(new Error('会话已切换'));
    });
    flushRequests();
  } else if (data.type === "xpeech:interaction-unavailable" && data.session_id === interactionSession) {
    interactionSession = null;
    rejectRequests('会话已切换或关闭', true);
  } else if (data.type === "xpeech:response") {
    var pending = pendingRequests.get(data.request_id);
    if (!pending || !pending.sent || data.session_id !== pending.sessionId) return;
    if (pending.data.method === 'newConversation' && typeof data.error !== 'string') {
      if (!data.result || typeof data.result.session_id !== 'string' || !data.result.session_id) return;
      if (data.result.session_id !== pending.sessionId) {
        pending.nextSessionId = data.result.session_id;
        interactionSession = null;
        rejectRequests('会话已切换', true, data.request_id);
        return;
      }
    }
    clearTimeout(pending.timer);
    pendingRequests.delete(data.request_id);
    if (typeof data.error === 'string') pending.reject(new Error(data.error));
    else pending.resolve(data.result);
    flushRequests();
  } else if (data.type === "xpeech:reply" && data.session_id === interactionSession && typeof data.content === 'string') {
    var reply = { session_id: data.session_id, content: data.content, request_id: data.request_id ?? null };
    replyListeners.forEach(function (listener) {
      try { listener(reply); } catch (error) { console.error('Xpeech onReply callback failed', error); }
    });
    window.dispatchEvent(new CustomEvent('xpeech:reply', { detail: reply }));
  } else if (data.type === 'xpeech:file' && data.session_id === interactionSession &&
      typeof data.path === 'string' && data.path.trim() &&
      data.request_id == null) {
    var file = Object.assign({ session_id: data.session_id, request_id: null }, fileInfo(data.path));
    chatFileListeners.forEach(function (listener) {
      try { listener(file); } catch (error) { console.error('Xpeech file callback failed', error); }
    });
    window.dispatchEvent(new CustomEvent('xpeech:chat-file', { detail: file }));
  } else if (data.type === "xpeech:expired") {
    refresh();
  } else if (data.type === "xpeech:error") {
    report(new Error(typeof data.message === "string" ? data.message : "登录失败"));
  }
}
async function connect(expectedVersion) {
  if (controller || (!currentJWT && !jwtProvider) || !host) return;
  var pending = new AbortController();
  controller = pending;
  try {
    if (jwtProvider) {
      var next = normalize(await jwtProvider());
      if (expectedVersion !== version || !host) return;
      if (currentId !== null && (next.userId !== currentId || next.channelName !== channelName)) {
        throw new Error("续期 JWT 不能切换用户或通道，请先调用 destroy()");
      }
      currentId = next.userId;
      channelName = next.channelName;
      currentJWT = next.token;
    }
    var response = await fetch(endpoint, {
      method: "GET", credentials: "omit",
      headers: { Authorization: "Bearer " + currentJWT },
      signal: pending.signal
    });
    var result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "JWT 校验失败");
    if (expectedVersion !== version || !host) return;
    if (result.user_id !== currentId || !Number.isFinite(result.expires_in) || result.expires_in <= 0) {
      throw new Error("无效的 JWT 校验响应");
    }
    verifiedJWT = currentJWT;
    needsRetry = false;
    createFrame();
    deliver();
    clearTimeout(refreshTimer);
    var remaining = result.expires_in;
    var delay = jwtProvider ? remaining - Math.min(30, remaining / 2) : remaining;
    refreshTimer = setTimeout(refresh, Math.min(2147483647, Math.max(1000, delay * 1000)));
  } catch (error) {
    if (expectedVersion === version && host && error.name !== "AbortError") report(error);
  } finally {
    if (controller === pending) controller = null;
  }
}
function refresh() {
  if ((currentJWT || jwtProvider) && host) connect(version);
}
function normalize(value) {
  if (typeof value !== "string" || !value.trim() || value.trim().length > 16384) {
    throw new TypeError("init 需要有效的 JWT 字符串");
  }
  var token = value.trim();
  try {
    var parts = token.split(".");
    if (parts.length !== 3 || parts.some(function (part) { return !/^[A-Za-z0-9_-]+$/.test(part); })) {
      throw new Error("invalid JWT");
    }
    function decode(part) {
      var encoded = part.replace(/-/g, "+").replace(/_/g, "/");
      var bytes = Uint8Array.from(atob(encoded), function (char) { return char.charCodeAt(0); });
      return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    }
    var header = decode(parts[0]);
    var claims = decode(parts[1]);
    if (!header || typeof header.kid !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/.test(header.kid) ||
        ["internal", "feishu", "web_client"].includes(header.kid) ||
        !claims || typeof claims.sub !== "string" || !claims.sub.trim()) throw new Error("invalid JWT identity");
    // These unverified fields only enforce the SDK lifecycle. The server verifies all claims.
    return { token: token, userId: claims.sub, channelName: header.kid };
  } catch (_) {
    throw new TypeError("JWT 格式无效，必须包含自定义通道 kid 和用户 sub");
  }
}
function ensureConnected() {
  if (host && (currentJWT || jwtProvider) && (!verifiedJWT || !frame || needsRetry) && !controller) {
    showStatus("正在连接…");
    if (verifiedJWT && !needsRetry) { createFrame(); deliver(); } else connect(version);
  }
}
function normalizeSenderName(sender) {
  if (typeof sender !== "string" || !sender.trim()) {
    throw new TypeError("senderName 必须为非空字符串");
  }
  return sender.trim();
}
function normalizeSessionMetadata(metadata) {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new TypeError("sessionMetadata 必须为 JSON 对象");
  }
  try {
    metadata = JSON.parse(JSON.stringify(metadata));
  } catch (_) {
    throw new TypeError("sessionMetadata 必须为 JSON 对象");
  }
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new TypeError("sessionMetadata 必须为 JSON 对象");
  }
  return metadata;
}
function init(value, senderName, title = "AI 助手", sessionMetadata = {}) {
  if (identityConfigured) throw new Error("init 只能初始化一次，请先调用 destroy() 再设置 JWT");
  var nextSenderName = normalizeSenderName(senderName);
  if (typeof title !== "string" || !title.trim()) throw new TypeError("title 必须为非空字符串");
  var nextMetadata = normalizeSessionMetadata(sessionMetadata);
  if (typeof value === "function") {
    identityConfigured = true;
    jwtProvider = value;
    currentSenderName = nextSenderName;
    currentTitle = title.trim();
    currentSessionMetadata = nextMetadata;
    initialize();
    showStatus("正在连接…");
    connect(version);
    return;
  }
  var next = normalize(value);
  currentSenderName = nextSenderName;
  currentTitle = title.trim();
  currentSessionMetadata = nextMetadata;
  identityConfigured = true;
  initialize();
  currentId = next.userId;
  channelName = next.channelName;
  currentJWT = next.token;
  jwtProvider = null;
  version += 1;
  clearTimeout(refreshTimer);
  if (controller) controller.abort();
  controller = null;
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
  verifiedJWT = null;
  currentId = null;
  currentJWT = null;
  jwtProvider = null;
  identityConfigured = false;
  currentSenderName = undefined;
  currentTitle = "AI 助手";
  currentSessionMetadata = {};
  channelName = null;
  needsRetry = false;
  removeFrame();
  replyListeners.clear();
  chatFileListeners.clear();
  window.removeEventListener("message", onMessage);
  window.removeEventListener("resize", keepInViewport);
  if (drag) endDrag({ pointerId: drag.id, type: "pointercancel" });
  if (resize) endResize({ pointerId: resize.id });
  panelSize = null;
  maximized = false;
  normalPanelBounds = null;
  suppressLauncherClick = false;
  if (host) host.remove();
  host = root = panel = launcher = closeButton = minimizeButton = maximizeButton = body = status = null;
}

export { init, hideButton, showButton, open, close, maximize, restore, minimize, destroy, setInput, addFiles, newConversation, send, chatSync, downloadFile, getFile, onReply, onChatFile };
