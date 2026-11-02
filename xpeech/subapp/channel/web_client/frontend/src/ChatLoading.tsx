/** Keep the embedded loading stages aligned with the welcome screen and composer. */
export default function ChatLoading() {
  return (
    <div className="chat-page chat-loading" role="status" aria-label="正在加载聊天" aria-busy="true">
      <div className="chat-content" aria-hidden="true">
        <div className="welcome-area chat-loading-welcome">
          <div className="chat-loading-intro">
            <span className="chat-loading-placeholder chat-loading-icon" />
            <div className="chat-loading-copy">
              <span className="chat-loading-placeholder chat-loading-title" />
              <span className="chat-loading-placeholder chat-loading-description" />
            </div>
          </div>
          <div className="chat-loading-prompts">
            <span className="chat-loading-placeholder" />
            <span className="chat-loading-placeholder" />
            <span className="chat-loading-placeholder" />
          </div>
        </div>
      </div>
      <div className="sender-wrap" aria-hidden="true">
        <div className="chat-loading-composer">
          <span className="chat-loading-placeholder chat-loading-input" />
          <span className="chat-loading-placeholder chat-loading-send" />
        </div>
        <div className="sender-footer"><span className="chat-loading-placeholder chat-loading-tip" /></div>
      </div>
    </div>
  );
}

export function EmbeddedLoading() {
  return (
    <main className="embedded-app">
      <section className="chat-workspace chat-workspace-embedded" aria-label="会话工作区">
        <div className="conversation-toolbar" aria-hidden="true">
          <span className="chat-loading-placeholder chat-loading-toolbar" />
        </div>
        <section className="conversation-panel"><ChatLoading /></section>
      </section>
    </main>
  );
}
