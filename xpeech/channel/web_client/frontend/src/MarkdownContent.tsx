import { Children } from 'react';
import { CodeHighlighter } from '@ant-design/x';
import type { ComponentProps, XMarkdownProps } from '@ant-design/x-markdown';
import { XMarkdown } from '@ant-design/x-markdown';

const languageAliases: Record<string, string> = {
  'c++': 'cpp',
  cs: 'csharp',
  html: 'markup',
  js: 'javascript',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  ts: 'typescript',
  yml: 'yaml',
};

function MarkdownCode({
  block,
  children,
  className,
  domNode: _domNode,
  lang,
  streamStatus: _streamStatus,
  ...rest
}: ComponentProps) {
  const code = Children.toArray(children).join('').replace(/\n$/, '');
  const requestedLanguage = lang?.trim().split(/\s+/, 1)[0].toLowerCase();
  const language = requestedLanguage
    ? languageAliases[requestedLanguage] || requestedLanguage
    : undefined;

  if (block && language) {
    return (
      <CodeHighlighter lang={language} prismLightMode={false}>
        {code}
      </CodeHighlighter>
    );
  }
  return <code className={className} {...rest}>{children}</code>;
}

const markdownComponents: NonNullable<XMarkdownProps['components']> = {
  code: MarkdownCode,
};

export default function MarkdownContent({ className, ...props }: XMarkdownProps) {
  return (
    <XMarkdown
      {...props}
      className={['x-markdown-light', 'markdown-content', className].filter(Boolean).join(' ')}
      components={markdownComponents}
      openLinksInNewTab
    />
  );
}
