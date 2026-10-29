import { Children, useEffect, useState } from 'react';
import type { ComponentProps, XMarkdownProps } from '@ant-design/x-markdown';
import { XMarkdown } from '@ant-design/x-markdown';
import Prism from 'prismjs/components/prism-core';
import 'prismjs/themes/prism.css';

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

const loadedLanguages = new Set<string>();
const languageLoaders: Record<string, () => Promise<unknown>> = {
  bash: () => import('prismjs/components/prism-bash'),
  clike: () => import('prismjs/components/prism-clike'),
  css: async () => {
    await loadLanguage('markup');
    return import('prismjs/components/prism-css');
  },
  c: async () => {
    await loadLanguage('clike');
    return import('prismjs/components/prism-c');
  },
  cpp: async () => {
    await loadLanguage('c');
    return import('prismjs/components/prism-cpp');
  },
  csharp: async () => {
    await loadLanguage('clike');
    return import('prismjs/components/prism-csharp');
  },
  java: async () => {
    await loadLanguage('clike');
    return import('prismjs/components/prism-java');
  },
  javascript: async () => {
    await loadLanguage('clike');
    return import('prismjs/components/prism-javascript');
  },
  json: async () => {
    await loadLanguage('javascript');
    return import('prismjs/components/prism-json');
  },
  markup: () => import('prismjs/components/prism-markup'),
  python: () => import('prismjs/components/prism-python'),
  sql: () => import('prismjs/components/prism-sql'),
  typescript: async () => {
    await loadLanguage('javascript');
    return import('prismjs/components/prism-typescript');
  },
  yaml: () => import('prismjs/components/prism-yaml'),
};

const loadingLanguages = new Map<string, Promise<boolean>>();

function loadLanguage(language: string): Promise<boolean> {
  if (loadedLanguages.has(language)) return Promise.resolve(true);
  const loader = languageLoaders[language];
  if (!loader) return Promise.resolve(false);
  const inFlight = loadingLanguages.get(language);
  if (inFlight) return inFlight;

  const promise = loader()
    .then(() => {
      loadedLanguages.add(language);
      return true;
    })
    .catch(() => false)
    .finally(() => loadingLanguages.delete(language));
  loadingLanguages.set(language, promise);
  return promise;
}

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
  const [highlightedLanguage, setHighlightedLanguage] = useState<string>();

  useEffect(() => {
    let active = true;
    setHighlightedLanguage(undefined);
    if (!block || !language) return () => { active = false; };
    void loadLanguage(language).then((loaded) => {
      if (active && loaded) setHighlightedLanguage(language);
    });
    return () => { active = false; };
  }, [block, language]);

  const grammar = highlightedLanguage ? Prism.languages[highlightedLanguage] : undefined;
  if (block && language && grammar) {
    return (
      <code
        className={[className, `language-${language}`].filter(Boolean).join(' ')}
        {...rest}
        dangerouslySetInnerHTML={{ __html: Prism.highlight(code, grammar, language) }}
      />
    );
  }
  return <code className={className} {...rest}>{block ? code : children}</code>;
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
