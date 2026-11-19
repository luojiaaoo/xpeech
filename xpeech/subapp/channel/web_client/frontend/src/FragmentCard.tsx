import { useEffect, useState } from 'react';
import { Alert, Button } from 'antd';
import type { ButtonProps } from 'antd';
import { activateFragment, requestFragmentCard } from './embedBridge';
import type { FragmentActionCard, FragmentButton } from './embedBridge';

function buttonAppearance(style: FragmentButton['style']): Pick<ButtonProps, 'type' | 'danger' | 'color' | 'variant'> {
  if (style === 'danger') return { type: 'default', danger: true };
  if (style === 'warning') return { color: 'orange', variant: 'outlined' };
  if (style === 'success') return { color: 'green', variant: 'outlined' };
  return { type: style };
}

export default function FragmentCard({ sessionId, revision, fragmentId, text }: {
  sessionId: string; revision: number; fragmentId: string; text: string;
}) {
  const [result, setResult] = useState<{ text: string; card: FragmentActionCard | null }>();

  useEffect(() => requestFragmentCard(
    sessionId, revision, fragmentId, text, (card) => setResult({ text, card }),
  ), [sessionId, revision, fragmentId, text]);

  if (result?.text !== text) return null;
  const { card } = result;
  if (!card) return <Alert type="error" showIcon message="片段卡片生成失败，请重试" />;
  if (!card.buttons.length) return null;

  return (
    <section className="chat-fragment-card" aria-label={card.title}>
      <strong>{card.title}</strong>
      {card.description.trim() ? <p className="chat-fragment-description">{card.description}</p> : null}
      <div className="chat-fragment-actions">
        {card.buttons.map((button, index) => (
          <Button key={index} size="small" {...buttonAppearance(button.style)}
            onClick={() => activateFragment(sessionId, revision, fragmentId, index)}>
            {button.label}
          </Button>
        ))}
      </div>
    </section>
  );
}
