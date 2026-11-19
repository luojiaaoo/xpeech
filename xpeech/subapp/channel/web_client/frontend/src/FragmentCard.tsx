import { useEffect, useState } from 'react';
import { Alert, Button } from 'antd';
import type { ButtonProps } from 'antd';
import { activateFragment, requestFragmentButtons } from './embedBridge';
import type { FragmentButton } from './embedBridge';

function buttonAppearance(style: FragmentButton['style']): Pick<ButtonProps, 'type' | 'danger' | 'color' | 'variant'> {
  if (style === 'danger') return { type: 'default', danger: true };
  if (style === 'warning') return { color: 'orange', variant: 'outlined' };
  if (style === 'success') return { color: 'green', variant: 'outlined' };
  return { type: style };
}

export default function FragmentCard({ sessionId, revision, fragmentId, text }: {
  sessionId: string; revision: number; fragmentId: string; text: string;
}) {
  const [result, setResult] = useState<{ text: string; buttons: FragmentButton[] | null }>();

  useEffect(() => requestFragmentButtons(
    sessionId, revision, fragmentId, text, (buttons) => setResult({ text, buttons }),
  ), [sessionId, revision, fragmentId, text]);

  if (result?.text !== text) return null;
  const { buttons } = result;
  if (!buttons) return <Alert type="error" showIcon message="片段按钮生成失败，请重试" />;
  if (!buttons.length) return null;

  return (
    <section className="chat-fragment-card" aria-label="片段操作">
      {buttons.map((button, index) => (
        <Button key={index} size="small" {...buttonAppearance(button.style)}
          onClick={() => activateFragment(sessionId, revision, fragmentId, index)}>
          {button.label}
        </Button>
      ))}
    </section>
  );
}
