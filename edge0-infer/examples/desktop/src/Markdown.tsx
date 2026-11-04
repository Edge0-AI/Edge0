import { Children, createContext, isValidElement, useContext, type ReactNode } from 'react';
import { Streamdown, type Components } from 'streamdown';
import { protectMath } from './chat';
import { Icon } from './Icon';
const CopyControls = createContext<{ disabled: boolean; copy: (text: string) => void | Promise<void> }>({ disabled: false, copy: text => navigator.clipboard.writeText(text) });
function CodeBlock({ children }: { children?: ReactNode }) {
  const { disabled, copy } = useContext(CopyControls);
  const code = Children.toArray(children).find(child => isValidElement<{ className?: string }>(child));
  const language = isValidElement<{ className?: string }>(code) ? code.props.className?.match(/language-(\S+)/)?.[1] : undefined;
  return <div className="code"><header>{language ?? 'code'}<span className="sp"/><button type="button" className="icon-btn" title="Copy code" disabled={disabled} onClick={() => void copy(textContent(children).replace(/\n$/, ''))}><Icon name="copy"/></button></header><pre>{children}</pre></div>;
}
const components: Components = {
  strong: ({ children }) => <strong>{children as ReactNode}</strong>,
  a: ({ href, children }) => <a href={typeof href === 'string' ? href : undefined} target="_blank" rel="noreferrer">{children as ReactNode}</a>,
  pre: ({ children }) => <CodeBlock>{children as ReactNode}</CodeBlock>,
};
function textContent(node: ReactNode): string { return Children.toArray(node).map(child => typeof child === 'string' || typeof child === 'number' ? String(child) : isValidElement<{ children?: ReactNode }>(child) ? textContent(child.props.children) : '').join(''); }
export function Markdown({ children, disabled = false, copy = text => navigator.clipboard.writeText(text) }: { children: string; disabled?: boolean; copy?: (text: string) => void | Promise<void> }) {
  return <CopyControls.Provider value={{ disabled, copy }}><Streamdown mode="static" controls={false} components={components}>{protectMath(children)}</Streamdown></CopyControls.Provider>;
}
