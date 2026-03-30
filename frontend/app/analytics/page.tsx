"use client";

import { useEffect, useState, useCallback } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

type StrategyMetric = { name: string; value: number };
type DashboardAverages = { avg_latency: number; avg_tokens: number; avg_trust: number; avg_risk: number; total_requests: number };
type TimelineMetric = { date: string; total: number; low_trust_count: number };
type RecentTrace = {
    id: string; query: string; strategy_selected: string; trust_score: number;
    latency_ms: number; verification_status?: string; verification_confidence?: number;
    difficulty_level?: string; created_at: string;
};
type AnalyticsResponse = {
    strategies: StrategyMetric[]; averages: DashboardAverages; timeline: TimelineMetric[];
    recent_traces: RecentTrace[]; verification_distribution?: StrategyMetric[];
    difficulty_distribution?: StrategyMetric[]; error?: string;
};

type ReasoningStep = { step_index: number; content: string; assumptions: string | string[]; flagged: boolean };
type TraceDetail = {
    request: {
        id: number; query: string; strategy_selected: string; hallucination_risk: number;
        confidence_score: number; trust_score: number; tokens_used: number;
        latency_ms: number; final_answer: string; verification_status?: string;
        verification_confidence?: number; difficulty_level?: string;
        retry_used?: boolean; created_at: string;
    };
    reasoning_steps: ReasoningStep[];
};

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
const VERIFICATION_COLORS: Record<string, string> = { PASSED: '#10b981', FAILED: '#ef4444', PARTIAL: '#f59e0b', HEURISTIC: '#3b82f6' };
const DIFFICULTY_COLORS: Record<string, string> = { EASY: '#38bdf8', MEDIUM: '#f59e0b', HARD: '#f43f5e' };

const BADGE_STYLES: Record<string, string> = {
    PASSED: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
    FAILED: 'bg-red-500/20 text-red-400 border border-red-500/30',
    PARTIAL: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
    HEURISTIC: 'bg-blue-500/20 text-blue-400 border border-blue-500/30',
    EASY: 'bg-sky-500/20 text-sky-400',
    MEDIUM: 'bg-amber-500/20 text-amber-400',
    HARD: 'bg-rose-500/20 text-rose-400',
};

export default function AnalyticsDashboard() {
    const [data, setData] = useState<AnalyticsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedTrace, setSelectedTrace] = useState<TraceDetail | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);
    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

    useEffect(() => {
        const controller = new AbortController();
        const loadAnalytics = async () => {
            try {
                setLoading(true); setError(null);
                const res = await fetch(`${apiBaseUrl}/api/v1/analytics`, { signal: controller.signal, cache: 'no-store' });
                if (!res.ok) throw new Error(`API Error: ${res.status} ${res.statusText}`);
                const payload: AnalyticsResponse = await res.json();
                if (payload.error) throw new Error(payload.error);
                setData(payload);
            } catch (err) {
                if ((err as Error).name === 'AbortError') return;
                setError((err as Error).message || 'Failed to load analytics.');
                setData(null);
            } finally { setLoading(false); }
        };
        loadAnalytics();
        return () => controller.abort();
    }, [apiBaseUrl]);

    const openTrace = useCallback(async (traceId: string) => {
        setTraceLoading(true); setSelectedTrace(null);
        try {
            const res = await fetch(`${apiBaseUrl}/api/v1/traces/${traceId}`, { cache: 'no-store' });
            if (!res.ok) throw new Error(`Error ${res.status}`);
            const detail: TraceDetail = await res.json();
            setSelectedTrace(detail);
        } catch (err) { console.error('Failed to load trace:', err); }
        finally { setTraceLoading(false); }
    }, [apiBaseUrl]);

    const closeTrace = () => setSelectedTrace(null);

    if (loading) return <div className="p-8 text-gray-400 text-center">Loading telemetry data...</div>;
    if (error) return <div className="p-8 text-red-400 text-center">Failed to load analytics: {error}</div>;
    if (!data) return <div className="p-8 text-red-400 text-center">No analytics data available.</div>;

    const strategies = data.strategies || [];
    const timeline = data.timeline || [];
    const recentTraces = data.recent_traces || [];
    const verificationDist = data.verification_distribution || [];
    const difficultyDist = data.difficulty_distribution || [];

    const trustColor = (score: number) =>
        score > 0.8 ? 'text-emerald-400' : score > 0.5 ? 'text-amber-400' : 'text-red-400';

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-8">
            <header>
                <h1 className="text-2xl font-bold text-gray-100" id="analytics-title">Fleet Telemetry</h1>
                <p className="text-gray-500 text-sm">Aggregated orchestration metrics — click a trace row to inspect</p>
            </header>

            {/* KPI Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4" id="kpi-cards">
                <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg">
                    <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Total Inferences</div>
                    <div className="text-2xl font-black">{data.averages?.total_requests || 0}</div>
                </div>
                <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg">
                    <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Avg Latency</div>
                    <div className="text-2xl font-black">{data.averages?.avg_latency || 0} ms</div>
                </div>
                <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg">
                    <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Avg Trust</div>
                    <div className={`text-2xl font-black ${trustColor(data.averages?.avg_trust || 0)}`}>
                        {((data.averages?.avg_trust || 0) * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg">
                    <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Avg Risk</div>
                    <div className="text-2xl font-black text-amber-400">{((data.averages?.avg_risk || 0) * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-900 border border-gray-800 p-4 rounded-lg">
                    <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Avg Tokens</div>
                    <div className="text-2xl font-black">{data.averages?.avg_tokens || 0}</div>
                </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6" id="charts-row">
                {/* Strategy Distribution */}
                <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg h-72">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Strategy Allocation</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <PieChart>
                            <Pie data={strategies} cx="50%" cy="50%" innerRadius={45} outerRadius={65} paddingAngle={4} dataKey="value">
                                {strategies.map((_, i) => <Cell key={`s-${i}`} fill={COLORS[i % COLORS.length]} />)}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', fontSize: 12 }} />
                            <Legend verticalAlign="bottom" height={24} iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Verification Distribution */}
                <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg h-72">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Verification Status</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <PieChart>
                            <Pie data={verificationDist} cx="50%" cy="50%" innerRadius={45} outerRadius={65} paddingAngle={4} dataKey="value">
                                {verificationDist.map((entry) => (
                                    <Cell key={`v-${entry.name}`} fill={VERIFICATION_COLORS[entry.name] || '#6b7280'} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', fontSize: 12 }} />
                            <Legend verticalAlign="bottom" height={24} iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Difficulty Distribution */}
                <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg h-72">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Difficulty Breakdown</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <PieChart>
                            <Pie data={difficultyDist} cx="50%" cy="50%" innerRadius={45} outerRadius={65} paddingAngle={4} dataKey="value">
                                {difficultyDist.map((entry) => (
                                    <Cell key={`d-${entry.name}`} fill={DIFFICULTY_COLORS[entry.name] || '#6b7280'} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', fontSize: 12 }} />
                            <Legend verticalAlign="bottom" height={24} iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Trust Timeline */}
                <div className="bg-gray-900 border border-gray-800 p-5 rounded-lg h-72">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Trust Timeline</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <BarChart data={timeline}>
                            <XAxis dataKey="date" stroke="#4B5563" fontSize={9} />
                            <YAxis stroke="#4B5563" fontSize={9} />
                            <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', fontSize: 12 }} />
                            <Bar dataKey="total" fill="#3b82f6" name="Total" radius={[2, 2, 0, 0]} />
                            <Bar dataKey="low_trust_count" fill="#ef4444" name="Low Trust" radius={[2, 2, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Recent Traces Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden" id="traces-table">
                <div className="p-4 border-b border-gray-800">
                    <h3 className="text-sm font-semibold text-gray-300">Recent Reasoning Traces</h3>
                    <p className="text-[10px] text-gray-500 mt-1">Click any row to inspect the full reasoning trace</p>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left text-gray-400">
                        <thead className="text-[10px] uppercase bg-gray-950 text-gray-500">
                            <tr>
                                <th className="px-5 py-3">Timestamp</th>
                                <th className="px-5 py-3">Query</th>
                                <th className="px-5 py-3">Strategy</th>
                                <th className="px-5 py-3">Difficulty</th>
                                <th className="px-5 py-3">Verification</th>
                                <th className="px-5 py-3">Trust</th>
                                <th className="px-5 py-3">Latency</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentTraces.map((trace) => (
                                <tr
                                    key={trace.id}
                                    onClick={() => openTrace(trace.id)}
                                    className="border-b border-gray-800 hover:bg-blue-900/15 cursor-pointer transition-colors"
                                >
                                    <td className="px-5 py-3 whitespace-nowrap text-xs">{new Date(trace.created_at).toLocaleString()}</td>
                                    <td className="px-5 py-3 truncate max-w-[200px] text-xs">{trace.query}</td>
                                    <td className="px-5 py-3">
                                        <span className="bg-blue-900/30 text-blue-400 px-1.5 py-0.5 rounded text-[10px]">{trace.strategy_selected}</span>
                                    </td>
                                    <td className="px-5 py-3">
                                        {trace.difficulty_level && (
                                            <span className={`px-1.5 py-0.5 rounded text-[10px] ${BADGE_STYLES[trace.difficulty_level] || 'text-gray-400'}`}>
                                                {trace.difficulty_level}
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-5 py-3">
                                        {trace.verification_status && (
                                            <span className={`px-1.5 py-0.5 rounded text-[10px] border ${BADGE_STYLES[trace.verification_status] || 'text-gray-400'}`}>
                                                {trace.verification_status}
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-5 py-3">
                                        <span className={`text-xs ${trustColor(trace.trust_score)}`}>
                                            {(trace.trust_score * 100).toFixed(1)}%
                                        </span>
                                    </td>
                                    <td className="px-5 py-3 text-xs">{trace.latency_ms}ms</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Trace Detail Panel */}
            {(selectedTrace || traceLoading) && (
                <div className="fixed inset-0 z-50 flex items-start justify-end" onClick={closeTrace}>
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
                    <div
                        className="relative z-10 w-full max-w-2xl h-screen bg-gray-950 border-l border-gray-800 overflow-y-auto shadow-2xl flex flex-col"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between p-5 border-b border-gray-800 sticky top-0 bg-gray-950 z-10">
                            <h2 className="text-base font-semibold text-gray-100">Trace Detail</h2>
                            <button onClick={closeTrace} className="text-gray-500 hover:text-gray-200 text-2xl leading-none">&times;</button>
                        </div>

                        {traceLoading && (
                            <div className="flex-1 flex items-center justify-center text-gray-400">Loading trace...</div>
                        )}

                        {selectedTrace && !traceLoading && (
                            <div className="p-5 space-y-5">
                                {/* Query */}
                                <div>
                                    <div className="text-[10px] uppercase text-gray-500 mb-2 tracking-wider">Query</div>
                                    <p className="text-gray-200 bg-gray-900 p-3 rounded-lg text-sm leading-relaxed">{selectedTrace.request.query}</p>
                                </div>

                                {/* Metrics Grid */}
                                <div className="grid grid-cols-3 gap-2">
                                    <div className="bg-gray-900 p-3 rounded-lg text-center">
                                        <div className="text-[10px] text-gray-500 mb-1">Strategy</div>
                                        <div className="text-[10px] font-bold text-blue-400">{selectedTrace.request.strategy_selected}</div>
                                    </div>
                                    <div className="bg-gray-900 p-3 rounded-lg text-center">
                                        <div className="text-[10px] text-gray-500 mb-1">Trust</div>
                                        <div className={`text-sm font-bold ${trustColor(selectedTrace.request.trust_score)}`}>
                                            {(selectedTrace.request.trust_score * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="bg-gray-900 p-3 rounded-lg text-center">
                                        <div className="text-[10px] text-gray-500 mb-1">Risk</div>
                                        <div className={`text-sm font-bold ${selectedTrace.request.hallucination_risk < 0.3 ? 'text-emerald-400' : selectedTrace.request.hallucination_risk < 0.7 ? 'text-amber-400' : 'text-red-400'}`}>
                                            {(selectedTrace.request.hallucination_risk * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    {selectedTrace.request.verification_status && (
                                        <div className="bg-gray-900 p-3 rounded-lg text-center">
                                            <div className="text-[10px] text-gray-500 mb-1">Verification</div>
                                            <span className={`text-[10px] px-1.5 py-0.5 rounded border ${BADGE_STYLES[selectedTrace.request.verification_status] || 'text-gray-400'}`}>
                                                {selectedTrace.request.verification_status}
                                            </span>
                                        </div>
                                    )}
                                    {selectedTrace.request.difficulty_level && (
                                        <div className="bg-gray-900 p-3 rounded-lg text-center">
                                            <div className="text-[10px] text-gray-500 mb-1">Difficulty</div>
                                            <span className={`text-[10px] px-1.5 py-0.5 rounded ${BADGE_STYLES[selectedTrace.request.difficulty_level] || 'text-gray-400'}`}>
                                                {selectedTrace.request.difficulty_level}
                                            </span>
                                        </div>
                                    )}
                                    <div className="bg-gray-900 p-3 rounded-lg text-center">
                                        <div className="text-[10px] text-gray-500 mb-1">Latency</div>
                                        <div className="text-sm font-bold">{selectedTrace.request.latency_ms} ms</div>
                                    </div>
                                </div>

                                {selectedTrace.request.retry_used && (
                                    <div className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 px-3 py-2 rounded-lg">
                                        ↻ This trace used failure recovery retry
                                    </div>
                                )}

                                {/* Final Answer */}
                                <div>
                                    <div className="text-[10px] uppercase text-gray-500 mb-2 tracking-wider">Verified Output</div>
                                    <p className="text-gray-200 bg-gray-900 p-3 rounded-lg text-sm leading-relaxed">{selectedTrace.request.final_answer}</p>
                                </div>

                                {/* Reasoning Steps */}
                                <div>
                                    <div className="text-[10px] uppercase text-gray-500 mb-3 tracking-wider">
                                        Reasoning Trace ({selectedTrace.reasoning_steps.length} steps)
                                    </div>
                                    <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                                        {selectedTrace.reasoning_steps.map((step) => {
                                            const assumptions: string[] = Array.isArray(step.assumptions)
                                                ? step.assumptions
                                                : typeof step.assumptions === 'string' && step.assumptions
                                                    ? (() => { try { return JSON.parse(step.assumptions); } catch { return []; } })()
                                                    : [];
                                            return (
                                                <div
                                                    key={step.step_index}
                                                    className={`flex gap-3 border-l-2 pl-3 py-2 ${step.flagged ? 'border-red-500 bg-red-900/10 rounded-r-lg' : 'border-gray-700'}`}
                                                >
                                                    <div className={`shrink-0 text-xs font-mono mt-0.5 ${step.flagged ? 'text-red-400' : 'text-gray-500'}`}>[{step.step_index}]</div>
                                                    <div className="space-y-1">
                                                        <p className={`text-sm leading-relaxed ${step.flagged ? 'text-red-200' : 'text-gray-300'}`}>{step.content}</p>
                                                        {assumptions.length > 0 && (
                                                            <div className="text-xs text-gray-500 bg-gray-900 px-2 py-1 rounded inline-block">
                                                                <span className="text-gray-400 font-semibold">Assumptions:</span> {assumptions.join(', ')}
                                                            </div>
                                                        )}
                                                        {step.flagged && (
                                                            <div className="text-xs font-bold text-red-500 uppercase tracking-wider">⚠ Verification issue</div>
                                                        )}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
