import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Sparkles, Loader2, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export const LegalChat = () => {
  const apiBase = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
  const sessionStorageKey = 'legal_chat_session_id';
  const schemaModeStorageKey = 'legal_chat_schema_mode';
  const initialAssistantMessage =
    "Hello. I am your Juris Lex AI assistant. How can I help you with your legal documents or research today?";
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: initialAssistantMessage }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resetNextSession, setResetNextSession] = useState(false);
  const [useSchemaIntakeMode, setUseSchemaIntakeMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem(schemaModeStorageKey) === 'true';
    } catch {
      return false;
    }
  });
  const [sessionId, setSessionId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(sessionStorageKey);
    } catch {
      return null;
    }
  });
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      if (sessionId) {
        localStorage.setItem(sessionStorageKey, sessionId);
      }
    } catch {
      // ignore storage errors in restricted environments
    }
  }, [sessionId]);

  useEffect(() => {
    try {
      localStorage.setItem(schemaModeStorageKey, String(useSchemaIntakeMode));
    } catch {
      // ignore storage errors
    }
  }, [useSchemaIntakeMode]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const endpoint = useSchemaIntakeMode ? `${apiBase}/schema_intake` : `${apiBase}/query`;
      const requestBody = useSchemaIntakeMode
        ? {
            user_input: userMessage,
            session_id: sessionId,
            reset_session: resetNextSession,
            llm_model: 'llama3.1:8b',
            llm_timeout_sec: 180,
          }
        : {
            query: userMessage,
            session_id: sessionId,
            enable_intake_mode: true,
            reset_session: resetNextSession,
            corpus: 'all',
            top_k: 12,
            generate_answer: true,
          };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      if (resetNextSession) {
        setResetNextSession(false);
      }
      const returnedSessionId = data?.meta?.session_id || data?.session_id || null;
      if (returnedSessionId && returnedSessionId !== sessionId) {
        setSessionId(returnedSessionId);
      }

      const answer = useSchemaIntakeMode
        ? (data?.text || 'I could not generate an intake response right now.')
        : (data?.answer || "I could not generate a direct answer right now, but I retrieved legal context for your question.");

      let assistantContent = answer;
      if (useSchemaIntakeMode) {
        const caseType = data?.case_type || 'unknown';
        const mode = data?.mode || 'question';
        assistantContent += `\n\n_Schema intake: case_type=${caseType}, mode=${mode}._`;
        if (Array.isArray(data?.missing_fields) && data.missing_fields.length > 0) {
          assistantContent += `\n\n_Missing fields: ${data.missing_fields.join(', ')}._`;
        }
      } else {
        const issueDomain = data?.meta?.issue_domain || 'unknown';
        const intakeMode = data?.meta?.intake_mode || 'unknown';
        if (issueDomain || intakeMode) {
          assistantContent += `\n\n_Context routing: domain=${issueDomain}, intake mode=${intakeMode}._`;
        }
        if (data?.meta?.llm_error) {
          assistantContent += `\n\n_Note: local LLM generation failed in backend (\`${data.meta.llm_error}\`)._`;
        }
        if (data?.meta?.validation?.applied && (data?.meta?.validation?.rejected_law_lines || 0) > 0) {
          assistantContent += `\n\n_Note: removed ${data.meta.validation.rejected_law_lines} unverified legal reference(s) during validation._`;
        }
        if (data?.meta?.missing_facts?.length) {
          assistantContent += `\n\n_Missing facts being collected: ${data.meta.missing_facts.join(', ')}._`;
        }
        if (typeof data?.meta?.confidence_score === 'number') {
          assistantContent += `\n\n_Confidence score: ${data.meta.confidence_score.toFixed(2)}._`;
        }
      }

      setMessages(prev => [...prev, { role: 'assistant', content: assistantContent }]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "I apologize, but I encountered an error processing your request. Please try again." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSession = () => {
    if (isLoading) return;
    setResetNextSession(true);
    setMessages([{ role: 'assistant', content: initialAssistantMessage }]);
    setInput('');
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-surface-container-low overflow-hidden">
      {/* Chat Header */}
      <div className="px-8 py-6 border-b border-outline-variant/10 bg-surface">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-headline font-bold text-primary">AI Legal Chat</h2>
            <p className="text-sm text-on-surface-variant">Consult with our specialized legal intelligence.</p>
          </div>
          <button
            onClick={() => setUseSchemaIntakeMode((v) => !v)}
            disabled={isLoading}
            className={`inline-flex items-center gap-2 rounded-xl border px-4 py-2 text-xs font-semibold uppercase tracking-wide disabled:opacity-50 ${
              useSchemaIntakeMode
                ? 'border-primary/60 bg-primary text-white'
                : 'border-outline-variant/30 bg-white text-on-surface hover:bg-surface-container-low'
            }`}
            title="Toggle schema-driven intake mode"
          >
            {useSchemaIntakeMode ? 'Schema Intake On' : 'Schema Intake Off'}
          </button>
          <button
            onClick={handleNewSession}
            disabled={isLoading}
            className="inline-flex items-center gap-2 rounded-xl border border-outline-variant/30 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-wide text-on-surface hover:bg-surface-container-low disabled:opacity-50"
            title="Start a fresh intake session"
          >
            <RotateCcw size={14} />
            New Session
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-8 space-y-8 no-scrollbar"
      >
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex max-w-[80%] space-x-4 ${msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : 'flex-row'}`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
                  msg.role === 'user' ? 'bg-secondary-container text-on-secondary-container' : 'bg-primary text-white'
                }`}>
                  {msg.role === 'user' ? <User size={20} /> : <Sparkles size={20} className="fill-white/20" />}
                </div>
                
                <div className={`p-6 rounded-2xl shadow-ambient ${
                  msg.role === 'user' 
                    ? 'bg-primary text-white' 
                    : 'bg-white text-on-surface'
                }`}>
                  <div className="markdown-body prose prose-sm max-w-none">
                    <Markdown>{msg.content}</Markdown>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex space-x-4">
              <div className="w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center">
                <Loader2 size={20} className="animate-spin" />
              </div>
              <div className="p-6 rounded-2xl bg-white shadow-ambient">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-primary/20 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-primary/20 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="w-2 h-2 bg-primary/20 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-8 bg-surface border-t border-outline-variant/10">
        <div className="max-w-4xl mx-auto relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a legal question or request a document summary..."
            className="w-full bg-surface-container-low border-none rounded-2xl p-6 pr-16 text-on-surface placeholder:text-on-surface-variant focus:ring-2 focus:ring-primary/20 resize-none h-24 shadow-inner"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="absolute right-4 bottom-4 p-3 bg-primary text-white rounded-xl hover:opacity-90 disabled:opacity-50 transition-all shadow-lg shadow-primary/20"
          >
            <Send size={20} />
          </button>
        </div>
        <p className="text-center text-[10px] text-on-surface-variant mt-4 font-bold uppercase tracking-widest">
          Juris Lex AI can provide information but is not a substitute for legal counsel.
        </p>
      </div>
    </div>
  );
};
