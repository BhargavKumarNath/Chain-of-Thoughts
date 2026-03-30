"use client";

import { useState, useEffect, useCallback, useRef } from 'react';
import {
    ReasoningResponse,
    ReasoningRequest,
    StrategyEnum,
    VerificationStatus,
    DifficultyLevel,
    TraceListItem,
} from '../types/reasoning';

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const STRATEGY_OPTIONS = Object.values(StrategyEnum);

const VERIFICATION_COLORS: Record<string, string> = {
    PASSED: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
    FAILED: 'bg-red-500/20 text-red-400 border-red-500/40',
    PARTIAL: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
    HEURISTIC: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
};

const DIFFICULTY_COLORS: Record<string, string> = {
    EASY: 'bg-sky-500/20 text-sky-400',
    MEDIUM: 'bg-amber-500/20 text-amber-400',
    HARD: 'bg-rose-500/20 text-rose-400',
};

/* ------------------------------------------------------------------ */
/*  Helper Components                                                  */
/* ------------------------------------------------------------------ */

function Badge({ text, className }: { text: string; className: string }) {
    return (
        <span className={`px-2 py-0.5 rounded text-[11px] font-semibold uppercase tracking-wider border ${className}`}>
            {text}
        </span>
    );
}

function Meter({ value, label, color }: { value: number; label: string; color: string }) {
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-gray-400">{label}</span>
                <span className={color}>{(value * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5">
                <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${color.includes('emerald') || color.includes('green') ? 'bg-emerald-500' : color.includes('amber') || color.includes('yellow') ? 'bg-amber-500' : color.includes('red') ? 'bg-red-500' : 'bg-blue-500'}`}
                    style={{ width: `${Math.min(value * 100, 100)}%` }}
                />
            </div>
        </div>
    );
}

/* ------------------------------------------------------------------ */
/*  Main Console Page                                                  */
/* ------------------------------------------------------------------ */

export default function ReasoningConsole() {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ReasoningResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Options
    const [forceStrategy, setForceStrategy] = useState<StrategyEnum | ''>('');
    const [debugMode, setDebugMode] = useState(false);

    // Trace history
    const [traceHistory, setTraceHistory] = useState<TraceListItem[]>([]);
    const [pinnedIds, setPinnedIds] = useState<Set<string>>(new Set());
    const [historySearch, setHistorySearch] = useState('');
    const [showHistory, setShowHistory] = useState(false);

    // Expandable steps
    const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
    const [showFullTrace, setShowFullTrace] = useState(false);
    const [showDebugLog, setShowDebugLog] = useState(false);
    const [showTrustBreakdown, setShowTrustBreakdown] = useState(false);

    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
    const traceRef = useRef<HTMLDivElement>(null);

    // Load pinned IDs from localStorage
    useEffect(() => {
        try {
            const stored = localStorage.getItem('reasonops_pinned');
            if (stored) setPinnedIds(new Set(JSON.parse(stored)));
        } catch { /* ignore */ }
    }, []);

    // Fetch trace history
    const fetchHistory = useCallback(async () => {
        try {
            const params = new URLSearchParams({ limit: '50', offset: '0' });
            if (historySearch) params.set('search', historySearch);
            const res = await fetch(`${apiBaseUrl}/api/v1/traces?${params}`, { cache: 'no-store' });
            if (res.ok) {
                const data = await res.json();
                setTraceHistory(data.traces || []);
            }
        } catch { /* ignore */ }
    }, [apiBaseUrl, historySearch]);

    useEffect(() => {
        if (showHistory) fetchHistory();
    }, [showHistory, fetchHistory]);

    // Submit query
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setExpandedSteps(new Set());
        setShowFullTrace(false);
        setShowDebugLog(false);

        const requestPayload: ReasoningRequest = {
            query,
            ...(forceStrategy ? { force_strategy: forceStrategy as StrategyEnum } : {}),
            ...(debugMode ? { debug: true } : {}),
        };

        try {
            const res = await fetch(`${apiBaseUrl}/api/v1/reason`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload),
            });

            if (!res.ok) {
                const detail = await res.text();
                throw new Error(`API Error ${res.status}: ${detail || res.statusText}`);
            }

            const data: ReasoningResponse = await res.json();
            setResult(data);
            // Refresh history after a short delay (background task needs time to write)
            setShowHistory(true);
            setTimeout(() => fetchHistory(), 800);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'An unknown error occurred.';
            setError(message);
        } finally {
            setLoading(false);
        }
    };

    // Trace management
    const deleteTrace = async (traceId: string) => {
        try {
            await fetch(`${apiBaseUrl}/api/v1/traces/${traceId}`, { method: 'DELETE' });
            fetchHistory();
        } catch { /* ignore */ }
    };

    const clearAllTraces = async () => {
        if (!confirm('Delete ALL reasoning traces? This cannot be undone.')) return;
        try {
            await fetch(`${apiBaseUrl}/api/v1/traces`, { method: 'DELETE' });
            setTraceHistory([]);
        } catch { /* ignore */ }
    };

    const togglePin = (id: string) => {
        setPinnedIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id); else next.add(id);
            localStorage.setItem('reasonops_pinned', JSON.stringify(Array.from(next)));
            return next;
        });
    };

    const exportTrace = () => {
        if (!result) return;
        const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `trace_${Date.now()}.json`; a.click();
        URL.revokeObjectURL(url);
    };

    const toggleStep = (idx: number) => {
        setExpandedSteps(prev => {
            const next = new Set(prev);
            if (next.has(idx)) next.delete(idx); else next.add(idx);
            return next;
        });
    };

    // Trust score color
    const trustColor = (score: number) =>
        score >= 0.85 ? 'text-emerald-400' : score >= 0.5 ? 'text-amber-400' : 'text-red-400';

    const riskColor = (risk: number) =>
        risk < 0.2 ? 'text-emerald-400' : risk < 0.5 ? 'text-amber-400' : 'text-red-400';

    return (
        <div className="min-h-screen bg-gray-950 text-gray-100 font-mono">
            <div className="flex">
                {/* ============== HISTORY SIDEBAR ============== */}
                {showHistory && (
                    <aside className="w-80 border-r border-gray-800 bg-gray-900/50 h-[calc(100vh-56px)] overflow-y-auto flex-shrink-0 flex flex-col">
                        <div className="p-4 border-b border-gray-800 space-y-3 flex-shrink-0">
                            <div className="flex items-center justify-between">
                                <h3 className="text-sm font-bold text-gray-300">Trace History</h3>
                                <button onClick={clearAllTraces} className="text-[10px] text-red-400 hover:text-red-300 transition-colors">Clear All</button>
                            </div>
                            <input
                                value={historySearch}
                                onChange={(e) => setHistorySearch(e.target.value)}
                                placeholder="Search traces..."
                                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500 transition-colors"
                            />
                        </div>
                        <div className="flex-1 overflow-y-auto">
                            {traceHistory.length === 0 && (
                                <p className="text-xs text-gray-500 p-4">No traces found.</p>
                            )}
                            {/* Pinned first, then rest */}
                            {[...traceHistory]
                                .sort((a, b) => (pinnedIds.has(b.id) ? 1 : 0) - (pinnedIds.has(a.id) ? 1 : 0))
                                .map(trace => (
                                <div key={trace.id} className={`p-3 border-b border-gray-800 hover:bg-gray-800/50 transition-colors group ${pinnedIds.has(trace.id) ? 'bg-blue-900/10' : ''}`}>
                                    <div className="flex items-start justify-between gap-2">
                                        <p className="text-xs text-gray-300 line-clamp-2 flex-1">{trace.query}</p>
                                        <div className="flex gap-1 items-center flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                                            <button onClick={() => togglePin(trace.id)} className="text-gray-500 hover:text-blue-400 text-xs" title={pinnedIds.has(trace.id) ? 'Unpin' : 'Pin'}>
                                                {pinnedIds.has(trace.id) ? '★' : '☆'}
                                            </button>
                                            <button onClick={() => deleteTrace(trace.id)} className="text-gray-500 hover:text-red-400 text-xs" title="Delete">✕</button>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2 mt-1.5">
                                        <span className="text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">{trace.strategy_selected}</span>
                                        {trace.verification_status && (
                                            <span className={`text-[10px] px-1.5 py-0.5 rounded border ${VERIFICATION_COLORS[trace.verification_status] || 'text-gray-400'}`}>
                                                {trace.verification_status}
                                            </span>
                                        )}
                                        <span className={`text-[10px] ${trustColor(trace.trust_score)}`}>
                                            {(trace.trust_score * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="text-[10px] text-gray-600 mt-1">{new Date(trace.created_at).toLocaleString()}</div>
                                </div>
                            ))}
                        </div>
                    </aside>
                )}

                {/* ============== MAIN CONTENT ============== */}
                <main className="flex-1 p-8 overflow-y-auto h-[calc(100vh-56px)]">
                    <div className="max-w-6xl mx-auto space-y-6">
                        {/* Header */}
                        <header className="flex items-center justify-between border-b border-gray-800 pb-4">
                            <div>
                                <h1 className="text-2xl font-bold text-blue-400" id="page-title">ReasonOps Console</h1>
                                <p className="text-gray-500 text-sm">Meta-reasoning orchestration layer</p>
                            </div>
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={() => setShowHistory(!showHistory)}
                                    className={`text-xs px-3 py-1.5 rounded border transition-colors ${showHistory ? 'bg-blue-600/20 border-blue-500/40 text-blue-400' : 'border-gray-700 text-gray-400 hover:text-blue-400 hover:border-blue-500/40'}`}
                                    id="toggle-history-btn"
                                >
                                    {showHistory ? 'Hide History' : 'Show History'}
                                </button>
                            </div>
                        </header>

                        {/* Query Form */}
                        <form onSubmit={handleSubmit} className="space-y-3" id="query-form">
                            <textarea
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Enter a complex query to test the reasoning engine..."
                                className="w-full bg-gray-900 border border-gray-700 rounded-lg p-4 text-gray-100 focus:outline-none focus:border-blue-500 transition-colors h-28 resize-none"
                                id="query-input"
                            />
                            <div className="flex items-center gap-4">
                                <button
                                    type="submit"
                                    disabled={loading || !query.trim()}
                                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 shadow-lg shadow-blue-600/20 hover:shadow-blue-500/30"
                                    id="submit-btn"
                                >
                                    {loading ? (
                                        <span className="flex items-center gap-2">
                                            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                            Reasoning...
                                        </span>
                                    ) : 'Generate'}
                                </button>

                                {/* Strategy override */}
                                <select
                                    value={forceStrategy}
                                    onChange={(e) => setForceStrategy(e.target.value as StrategyEnum | '')}
                                    className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-300 focus:outline-none focus:border-blue-500 transition-colors"
                                    id="strategy-select"
                                >
                                    <option value="">Auto Strategy</option>
                                    {STRATEGY_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
                                </select>

                                {/* Debug toggle */}
                                <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer select-none">
                                    <input
                                        type="checkbox"
                                        checked={debugMode}
                                        onChange={(e) => setDebugMode(e.target.checked)}
                                        className="accent-blue-500"
                                        id="debug-toggle"
                                    />
                                    Debug Mode
                                </label>

                                {/* Export */}
                                {result && (
                                    <button
                                        type="button"
                                        onClick={exportTrace}
                                        className="text-xs text-gray-500 hover:text-blue-400 transition-colors ml-auto"
                                        id="export-btn"
                                    >
                                        Export JSON
                                    </button>
                                )}
                            </div>
                        </form>

                        {/* Error */}
                        {error && (
                            <div className="bg-red-900/30 border border-red-500/40 text-red-300 p-4 rounded-lg animate-fadeIn" id="error-display">
                                {error}
                            </div>
                        )}

                        {/* ============== RESULTS ============== */}
                        {result && (
                            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-fadeIn">
                                {/* LEFT: Telemetry Panel */}
                                <div className="lg:col-span-1 space-y-4">

                                    {/* Verification Status - Big Badge */}
                                    <div className={`p-4 rounded-lg border text-center ${VERIFICATION_COLORS[result.verification_status] || 'border-gray-700'}`} id="verification-badge">
                                        <div className="text-[10px] uppercase tracking-widest mb-1 opacity-70">
                                            {result.verification_status === 'HEURISTIC' ? 'Evaluation' : 'Verification'}
                                        </div>
                                        <div className="text-xl font-black">
                                            {result.verification_status === 'HEURISTIC' ? 'EVALUATED' : result.verification_status}
                                        </div>
                                        <div className="text-xs mt-1 opacity-80">{(result.verification_confidence * 100).toFixed(1)}% confidence</div>
                                    </div>

                                    {/* Strategy + Difficulty */}
                                    <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg space-y-3" id="strategy-panel">
                                        <div className="flex items-center gap-2">
                                            <Badge text={result.strategy_selected} className="bg-blue-500/20 text-blue-400 border-blue-500/40" />
                                            <Badge text={result.difficulty_level} className={DIFFICULTY_COLORS[result.difficulty_level] || 'bg-gray-700 text-gray-400'} />
                                        </div>
                                        {result.retry_used && result.retry_strategy && (
                                            <div className="text-[10px] text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1">
                                                ↻ Retried with {result.retry_strategy}
                                            </div>
                                        )}
                                    </div>

                                    {/* Trust Score */}
                                    <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg" id="trust-panel">
                                        <div className="flex items-center justify-between mb-3">
                                            <h3 className="text-gray-500 text-[10px] uppercase tracking-wider">Trust Score</h3>
                                            <button onClick={() => setShowTrustBreakdown(!showTrustBreakdown)} className="text-[10px] text-blue-400 hover:text-blue-300">
                                                {showTrustBreakdown ? 'Hide' : 'Details'}
                                            </button>
                                        </div>
                                        <div className={`text-3xl font-black ${trustColor(result.trust_score.aggregate_score)}`}>
                                            {(result.trust_score.aggregate_score * 100).toFixed(1)}%
                                        </div>

                                        {showTrustBreakdown && (
                                            <div className="mt-3 space-y-2 border-t border-gray-800 pt-3">
                                                <Meter value={result.trust_score.verification_confidence} label="Verification (50%)" color={trustColor(result.trust_score.verification_confidence)} />
                                                <Meter value={result.trust_score.reasoning_consistency_score} label="Consistency (25%)" color={trustColor(result.trust_score.reasoning_consistency_score)} />
                                                <Meter value={result.trust_score.self_consistency_score} label="Self-Consistency (25%)" color={trustColor(result.trust_score.self_consistency_score)} />
                                            </div>
                                        )}
                                    </div>

                                    {/* Hallucination Risk */}
                                    <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg" id="risk-panel">
                                        <h3 className="text-gray-500 text-[10px] uppercase tracking-wider mb-2">Hallucination Risk</h3>
                                        <div className={`text-2xl font-black ${riskColor(result.hallucination_risk)}`}>
                                            {(result.hallucination_risk * 100).toFixed(1)}%
                                        </div>
                                        <Meter value={result.hallucination_risk} label="" color={riskColor(result.hallucination_risk)} />
                                    </div>

                                    {/* Execution Telemetry */}
                                    <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg space-y-2 text-sm" id="telemetry-panel">
                                        <h3 className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Execution</h3>
                                        <div className="flex justify-between">
                                            <span className="text-gray-500">Latency</span>
                                            <span className="text-gray-200">{result.latency_ms} ms</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-500">Tokens</span>
                                            <span className="text-gray-200">{result.tokens_used}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-500">Steps</span>
                                            <span className="text-gray-200">{result.reasoning_steps.length}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* RIGHT: Output & Trace */}
                                <div className="lg:col-span-3 space-y-4">

                                    {/* Final Answer */}
                                    <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg" id="final-answer">
                                        <h3 className="text-gray-500 text-[10px] uppercase tracking-wider mb-3">Verified Output</h3>
                                        <p className="text-lg leading-relaxed text-gray-200">{result.final_answer}</p>
                                    </div>

                                    {/* Verification Details */}
                                    {result.verification_details && result.verification_details.length > 0 && (
                                        <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg" id="verification-details">
                                            <h3 className="text-gray-500 text-[10px] uppercase tracking-wider mb-3">
                                                Verification Checks ({result.verification_details.length})
                                            </h3>
                                            <div className="space-y-2 max-h-48 overflow-y-auto">
                                                {result.verification_details.map((d, i) => (
                                                    <div key={i} className={`flex items-start gap-2 text-xs p-2 rounded border ${d.passed ? 'border-emerald-500/20 bg-emerald-500/5' : 'border-red-500/20 bg-red-500/5'}`}>
                                                        <span className={`mt-0.5 flex-shrink-0 ${d.passed ? 'text-emerald-400' : 'text-red-400'}`}>
                                                            {d.passed ? '✓' : '✗'}
                                                        </span>
                                                        <div>
                                                            <span className="text-gray-400 font-medium">[{d.method}]</span>
                                                            <span className="text-gray-300 ml-1">{d.message}</span>
                                                            <span className="text-gray-600 ml-2">({(d.confidence * 100).toFixed(0)}%)</span>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Reasoning Trace */}
                                    <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg" id="reasoning-trace">
                                        <div className="flex items-center justify-between mb-4">
                                            <h3 className="text-gray-500 text-[10px] uppercase tracking-wider">
                                                Reasoning Trace ({result.reasoning_steps.length} steps)
                                            </h3>
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => {
                                                        if (expandedSteps.size === result.reasoning_steps.length) {
                                                            setExpandedSteps(new Set());
                                                        } else {
                                                            setExpandedSteps(new Set(result.reasoning_steps.map(s => s.step_index)));
                                                        }
                                                    }}
                                                    className="text-[10px] text-gray-500 hover:text-blue-400 transition-colors"
                                                >
                                                    {expandedSteps.size === result.reasoning_steps.length ? 'Collapse All' : 'Expand All'}
                                                </button>
                                                <button onClick={() => setShowFullTrace(true)} className="text-[10px] text-gray-500 hover:text-blue-400 transition-colors">
                                                    View Full Trace
                                                </button>
                                            </div>
                                        </div>

                                        <div ref={traceRef} className="space-y-2 max-h-[500px] overflow-y-auto pr-2 scrollbar-thin">
                                            {result.reasoning_steps.map((step) => {
                                                const isExpanded = expandedSteps.has(step.step_index);
                                                return (
                                                    <div
                                                        key={step.step_index}
                                                        className={`border-l-2 rounded-r-lg transition-all duration-200 ${step.flagged ? 'border-red-500 bg-red-900/10' : 'border-gray-700 hover:border-blue-500/50'}`}
                                                    >
                                                        <button
                                                            onClick={() => toggleStep(step.step_index)}
                                                            className="w-full flex items-start gap-3 p-3 text-left"
                                                        >
                                                            <span className={`text-xs font-mono flex-shrink-0 mt-0.5 ${step.flagged ? 'text-red-400' : 'text-gray-500'}`}>
                                                                [{step.step_index}]
                                                            </span>
                                                            <span className={`text-sm flex-1 ${step.flagged ? 'text-red-200' : 'text-gray-300'} ${isExpanded ? '' : 'line-clamp-2'}`}>
                                                                {step.content}
                                                            </span>
                                                            <span className="text-gray-600 text-xs flex-shrink-0">
                                                                {isExpanded ? '▾' : '▸'}
                                                            </span>
                                                        </button>

                                                        {isExpanded && (
                                                            <div className="px-3 pb-3 pl-10 space-y-2 animate-fadeIn">
                                                                {step.assumptions && step.assumptions.length > 0 && (
                                                                    <div className="text-xs text-gray-500 bg-gray-950 p-2 rounded border border-gray-800">
                                                                        <span className="font-semibold text-gray-400">Assumptions:</span> {step.assumptions.join(', ')}
                                                                    </div>
                                                                )}
                                                                {step.flagged && (
                                                                    <div className="text-xs font-bold text-red-500 uppercase tracking-wider flex items-center gap-1">
                                                                        <span>⚠</span> Verification issue detected
                                                                    </div>
                                                                )}
                                                                {step.verification_note && (
                                                                    <div className="text-xs text-amber-400 bg-amber-500/10 p-2 rounded">{step.verification_note}</div>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    {/* Debug Log */}
                                    {result.debug_log && result.debug_log.length > 0 && (
                                        <div className="bg-gray-900 border border-gray-800 rounded-lg" id="debug-log">
                                            <button
                                                onClick={() => setShowDebugLog(!showDebugLog)}
                                                className="w-full p-4 flex items-center justify-between text-left"
                                            >
                                                <h3 className="text-gray-500 text-[10px] uppercase tracking-wider">
                                                    Debug Log ({result.debug_log.length} entries)
                                                </h3>
                                                <span className="text-gray-600 text-xs">{showDebugLog ? '▾' : '▸'}</span>
                                            </button>
                                            {showDebugLog && (
                                                <div className="px-4 pb-4">
                                                    <pre className="text-xs text-gray-400 bg-gray-950 p-3 rounded border border-gray-800 max-h-64 overflow-y-auto font-mono leading-relaxed">
                                                        {result.debug_log.join('\n')}
                                                    </pre>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </main>
            </div>

            {/* ============== FULL TRACE MODAL ============== */}
            {showFullTrace && result && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-8" onClick={() => setShowFullTrace(false)}>
                    <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
                    <div
                        className="relative z-10 w-full max-w-4xl max-h-[85vh] bg-gray-950 border border-gray-800 rounded-xl shadow-2xl flex flex-col"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between p-5 border-b border-gray-800 flex-shrink-0">
                            <h2 className="text-base font-bold text-gray-100">Full Reasoning Trace</h2>
                            <button onClick={() => setShowFullTrace(false)} className="text-gray-500 hover:text-gray-200 text-2xl leading-none">&times;</button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-5 space-y-4">
                            {result.reasoning_steps.map((step) => (
                                <div key={step.step_index} className={`border-l-2 pl-4 py-2 ${step.flagged ? 'border-red-500 bg-red-900/10 rounded-r-md' : 'border-gray-700'}`}>
                                    <div className="flex items-start gap-3">
                                        <span className={`text-xs font-mono flex-shrink-0 ${step.flagged ? 'text-red-400' : 'text-gray-500'}`}>[{step.step_index}]</span>
                                        <div className="space-y-2 flex-1">
                                            <p className={`text-sm leading-relaxed ${step.flagged ? 'text-red-200' : 'text-gray-300'}`}>{step.content}</p>
                                            {step.assumptions && step.assumptions.length > 0 && (
                                                <div className="text-xs text-gray-500 bg-gray-900 p-2 rounded border border-gray-800">
                                                    <span className="font-semibold text-gray-400">Assumptions:</span> {step.assumptions.join(', ')}
                                                </div>
                                            )}
                                            {step.flagged && (
                                                <div className="text-xs font-bold text-red-500 uppercase">⚠ Verification issue</div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Global styles for animations */}
            <style jsx global>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(4px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
                .line-clamp-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
                .scrollbar-thin::-webkit-scrollbar { width: 4px; }
                .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
                .scrollbar-thin::-webkit-scrollbar-thumb { background: #374151; border-radius: 9999px; }
                .scrollbar-thin::-webkit-scrollbar-thumb:hover { background: #4B5563; }
            `}</style>
        </div>
    );
}
