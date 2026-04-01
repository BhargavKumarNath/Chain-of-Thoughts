"use client";

import { useEffect, useState, useCallback } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
    LineChart, Line, CartesianGrid,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface PolicyWeight {
    strategy: string;
    expected_reward: number;
    avg_trust: number;
    avg_tokens: number;
    avg_latency: number;
    samples: number;
    trust_stddev: number;
}

interface PolicyData {
    weights: Record<string, PolicyWeight>;
    last_optimized: string | null;
    lambda_cost: number;
    lambda_latency: number;
}

interface RewardPoint { date: string; reward: number; trust: number; tokens: number; latency: number; n: number }
interface LatencyDist { strategy: string; n: number; avg: number; min: number; max: number; p50: number; p90: number; p99: number; stddev: number }
interface StratPerf { strategy: string; n: number; avg_trust: number; avg_tokens: number; avg_latency: number; avg_risk: number; avg_reward: number }

interface ABArm {
    label: string; strategy: string;
    avg_trust: number; avg_tokens: number; avg_latency: number; avg_reward: number; total_tokens: number;
}
interface ABTest {
    sample_size: number;
    arms: { always_cot: ABArm; adaptive: ABArm; hybrid: ABArm };
    token_savings: { saved_vs_always_cot: number; savings_pct: number };
    error?: string;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const STRATEGY_COLORS: Record<string, string> = {
    DIRECT: '#38bdf8', SHORT_COT: '#10b981', LONG_COT: '#f59e0b',
    MULTI_SAMPLE: '#8b5cf6', TREE_OF_THOUGHTS: '#ec4899', EXTERNAL_SOLVER: '#ef4444',
};
const LINE_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#ec4899'];
const RISK_LABELS: Record<string, string> = { LOW_RISK: '🟢 Low Risk', MED_RISK: '🟡 Med Risk', HIGH_RISK: '🔴 High Risk' };

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function PolicyEnginePage() {
    const [policy, setPolicy] = useState<PolicyData | null>(null);
    const [rewardCurves, setRewardCurves] = useState<Record<string, RewardPoint[]>>({});
    const [abTest, setAbTest] = useState<ABTest | null>(null);
    const [latencyDist, setLatencyDist] = useState<LatencyDist[]>([]);
    const [stratPerf, setStratPerf] = useState<StratPerf[]>([]);
    const [loading, setLoading] = useState(true);
    const [optimizing, setOptimizing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const api = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

    const fetchAll = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [pRes, rRes, aRes, lRes, sRes] = await Promise.all([
                fetch(`${api}/api/v1/policy/weights`, { cache: 'no-store' }),
                fetch(`${api}/api/v1/policy/reward-curves`, { cache: 'no-store' }),
                fetch(`${api}/api/v1/policy/ab-test`, { cache: 'no-store' }),
                fetch(`${api}/api/v1/policy/latency`, { cache: 'no-store' }),
                fetch(`${api}/api/v1/policy/strategy-performance`, { cache: 'no-store' }),
            ]);
            const [pData, rData, aData, lData, sData] = await Promise.all([
                pRes.json(), rRes.json(), aRes.json(), lRes.json(), sRes.json(),
            ]);
            setPolicy(pData);
            setRewardCurves(rData.curves || {});
            setAbTest(aData);
            setLatencyDist(lData.distributions || []);
            setStratPerf(sData.strategies || []);
        } catch (err) {
            setError((err as Error).message || 'Failed to load policy data');
        } finally {
            setLoading(false);
        }
    }, [api]);

    useEffect(() => { fetchAll(); }, [fetchAll]);

    const handleOptimize = async () => {
        setOptimizing(true);
        try {
            const res = await fetch(`${api}/api/v1/policy/optimize`, { method: 'POST' });
            if (!res.ok) throw new Error(`Optimization failed: ${res.status}`);
            await fetchAll();
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setOptimizing(false);
        }
    };

    if (loading) return <div className="p-8 text-gray-400 text-center">Loading policy engine...</div>;
    if (error) return <div className="p-8 text-red-400 text-center">Error: {error}</div>;

    // Build unified reward curve data for LineChart
    const allDates = new Set<string>();
    Object.values(rewardCurves).forEach(pts => pts.forEach(p => allDates.add(p.date)));
    const sortedDates = Array.from(allDates).sort();
    const lineData = sortedDates.map(date => {
        const point: Record<string, any> = { date };
        Object.entries(rewardCurves).forEach(([strategy, pts]) => {
            const match = pts.find(p => p.date === date);
            if (match) point[strategy] = match.reward;
        });
        return point;
    });
    const strategiesInCurves = Object.keys(rewardCurves);

    // A/B chart data
    const abChartData = abTest && !abTest.error ? [
        { name: 'Trust', always_cot: abTest.arms.always_cot.avg_trust, adaptive: abTest.arms.adaptive.avg_trust, hybrid: abTest.arms.hybrid.avg_trust },
        { name: 'Reward', always_cot: abTest.arms.always_cot.avg_reward, adaptive: abTest.arms.adaptive.avg_reward, hybrid: abTest.arms.hybrid.avg_reward },
    ] : [];

    // Radar data for strategy performance
    const radarData = stratPerf.map(s => ({
        strategy: s.strategy,
        trust: s.avg_trust * 100,
        reward: Math.max(0, (s.avg_reward + 0.5) * 100),  // normalize to 0-100 range
        efficiency: Math.max(0, 100 - (s.avg_tokens / 20)),
        speed: Math.max(0, 100 - (s.avg_latency / 300)),
    }));

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-8">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-100" id="policy-title">Policy Engine</h1>
                    <p className="text-gray-500 text-sm">Contextual bandit optimization &amp; adaptive routing</p>
                </div>
                <button
                    onClick={handleOptimize}
                    disabled={optimizing}
                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-5 py-2 rounded-lg font-medium transition-all shadow-lg shadow-blue-600/20"
                    id="optimize-btn"
                >
                    {optimizing ? (
                        <span className="flex items-center gap-2">
                            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Optimizing...
                        </span>
                    ) : '⚡ Optimize Policy'}
                </button>
            </header>

            {/* ============ POLICY WEIGHTS / DECISION SURFACE ============ */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4" id="decision-surface">
                {policy && Object.entries(policy.weights).map(([bin, w]) => (
                    <div key={bin} className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-3">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-bold text-gray-300">{RISK_LABELS[bin] || bin}</h3>
                            <span className="text-[10px] text-gray-600">{w.samples} samples</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span
                                className="px-2.5 py-1 rounded text-xs font-bold"
                                style={{ backgroundColor: (STRATEGY_COLORS[w.strategy] || '#6b7280') + '30', color: STRATEGY_COLORS[w.strategy] || '#9ca3af' }}
                            >
                                {w.strategy}
                            </span>
                            <span className="text-xs text-gray-500">optimal</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="bg-gray-950 rounded p-2">
                                <div className="text-gray-500">Reward</div>
                                <div className={`font-bold ${w.expected_reward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {w.expected_reward >= 0 ? '+' : ''}{w.expected_reward.toFixed(4)}
                                </div>
                            </div>
                            <div className="bg-gray-950 rounded p-2">
                                <div className="text-gray-500">Trust</div>
                                <div className="font-bold text-blue-400">{(w.avg_trust * 100).toFixed(1)}%</div>
                            </div>
                            <div className="bg-gray-950 rounded p-2">
                                <div className="text-gray-500">Avg Tokens</div>
                                <div className="font-bold text-gray-300">{Math.round(w.avg_tokens)}</div>
                            </div>
                            <div className="bg-gray-950 rounded p-2">
                                <div className="text-gray-500">Avg Latency</div>
                                <div className="font-bold text-gray-300">{Math.round(w.avg_latency)}ms</div>
                            </div>
                        </div>
                    </div>
                ))}
            </section>

            {/* Optimization metadata */}
            {policy && (
                <div className="flex items-center gap-6 text-xs text-gray-500 bg-gray-900/50 border border-gray-800 rounded-lg px-5 py-3">
                    <span>λ_cost = {policy.lambda_cost}</span>
                    <span>λ_latency = {policy.lambda_latency}</span>
                    <span>R = Trust − λ·Cost − λ·Latency</span>
                    <span className="ml-auto">
                        Last optimized: {policy.last_optimized ? new Date(policy.last_optimized).toLocaleString() : 'Never'}
                    </span>
                </div>
            )}

            {/* ============ CHARTS ROW 1 ============ */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Reward Curves */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5" id="reward-curves">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Reward Curves Over Time</h3>
                    {lineData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={260}>
                            <LineChart data={lineData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                                <XAxis dataKey="date" stroke="#4B5563" fontSize={10} />
                                <YAxis stroke="#4B5563" fontSize={10} />
                                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', fontSize: 12 }} />
                                <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                                {strategiesInCurves.map((strategy, i) => (
                                    <Line
                                        key={strategy}
                                        type="monotone"
                                        dataKey={strategy}
                                        stroke={STRATEGY_COLORS[strategy] || LINE_COLORS[i % LINE_COLORS.length]}
                                        strokeWidth={2}
                                        dot={{ r: 3 }}
                                        connectNulls
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-64 flex items-center justify-center text-gray-600 text-sm">No reward data yet. Run queries to generate data.</div>
                    )}
                </div>

                {/* A/B Test Results */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5" id="ab-test">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">
                        A/B Test: Always-CoT vs Adaptive vs Hybrid
                    </h3>
                    {abTest && !abTest.error ? (
                        <div className="space-y-4">
                            <ResponsiveContainer width="100%" height={180}>
                                <BarChart data={abChartData} barCategoryGap="20%">
                                    <XAxis dataKey="name" stroke="#4B5563" fontSize={11} />
                                    <YAxis stroke="#4B5563" fontSize={10} />
                                    <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', fontSize: 12 }} />
                                    <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                                    <Bar dataKey="always_cot" name="Always CoT" fill="#ef4444" radius={[3, 3, 0, 0]} />
                                    <Bar dataKey="adaptive" name="Adaptive" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                                    <Bar dataKey="hybrid" name="Hybrid" fill="#10b981" radius={[3, 3, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                            {/* Token savings badge */}
                            <div className="flex items-center gap-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
                                <div className="text-emerald-400 text-2xl font-black">{abTest.token_savings.savings_pct}%</div>
                                <div>
                                    <div className="text-emerald-400 text-xs font-semibold">Token Cost Savings vs Always-CoT</div>
                                    <div className="text-gray-500 text-[10px]">
                                        {Math.round(abTest.token_savings.saved_vs_always_cot).toLocaleString()} tokens saved across {abTest.sample_size} requests
                                    </div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
                            {abTest?.error || 'Run queries to generate A/B data.'}
                        </div>
                    )}
                </div>
            </div>

            {/* ============ CHARTS ROW 2 ============ */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Latency Distribution */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5" id="latency-dist">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Latency Distribution by Strategy</h3>
                    {latencyDist.length > 0 ? (
                        <div className="space-y-3">
                            {latencyDist.map(d => {
                                const maxVal = Math.max(...latencyDist.map(x => x.p99));
                                return (
                                    <div key={d.strategy} className="space-y-1">
                                        <div className="flex items-center justify-between text-xs">
                                            <span
                                                className="font-bold px-1.5 py-0.5 rounded text-[10px]"
                                                style={{ backgroundColor: (STRATEGY_COLORS[d.strategy] || '#6b7280') + '30', color: STRATEGY_COLORS[d.strategy] || '#9ca3af' }}
                                            >
                                                {d.strategy}
                                            </span>
                                            <span className="text-gray-500">{d.n} reqs</span>
                                        </div>
                                        {/* Visual bar showing p50 / p90 / p99 */}
                                        <div className="relative h-6 bg-gray-950 rounded-md overflow-hidden">
                                            {/* p99 range */}
                                            <div
                                                className="absolute top-0 h-full rounded-md opacity-20"
                                                style={{ left: `${(d.min / maxVal) * 100}%`, width: `${((d.p99 - d.min) / maxVal) * 100}%`, backgroundColor: STRATEGY_COLORS[d.strategy] || '#6b7280' }}
                                            />
                                            {/* p90 range */}
                                            <div
                                                className="absolute top-0 h-full rounded-md opacity-40"
                                                style={{ left: `${(d.min / maxVal) * 100}%`, width: `${((d.p90 - d.min) / maxVal) * 100}%`, backgroundColor: STRATEGY_COLORS[d.strategy] || '#6b7280' }}
                                            />
                                            {/* p50 marker */}
                                            <div
                                                className="absolute top-0 h-full rounded-md opacity-80"
                                                style={{ left: `${(d.min / maxVal) * 100}%`, width: `${((d.p50 - d.min) / maxVal) * 100}%`, backgroundColor: STRATEGY_COLORS[d.strategy] || '#6b7280' }}
                                            />
                                            {/* Avg marker line */}
                                            <div
                                                className="absolute top-0 h-full w-0.5 bg-white/60"
                                                style={{ left: `${(d.avg / maxVal) * 100}%` }}
                                            />
                                        </div>
                                        <div className="flex gap-4 text-[10px] text-gray-500">
                                            <span>p50: {Math.round(d.p50)}ms</span>
                                            <span>p90: {Math.round(d.p90)}ms</span>
                                            <span>p99: {Math.round(d.p99)}ms</span>
                                            <span>avg: {Math.round(d.avg)}ms</span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">No latency data yet.</div>
                    )}
                </div>

                {/* Strategy Performance Radar */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5" id="strategy-perf">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Strategy Performance</h3>
                    {stratPerf.length > 0 ? (
                        <div className="space-y-4">
                            {/* Performance table */}
                            <div className="overflow-x-auto">
                                <table className="w-full text-xs text-left text-gray-400">
                                    <thead className="text-[10px] uppercase bg-gray-950 text-gray-500">
                                        <tr>
                                            <th className="px-3 py-2">Strategy</th>
                                            <th className="px-3 py-2">N</th>
                                            <th className="px-3 py-2">Trust</th>
                                            <th className="px-3 py-2">Tokens</th>
                                            <th className="px-3 py-2">Latency</th>
                                            <th className="px-3 py-2">Risk</th>
                                            <th className="px-3 py-2">Reward</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {stratPerf.map((s, i) => (
                                            <tr key={s.strategy} className={`border-b border-gray-800 ${i === 0 ? 'bg-emerald-900/10' : ''}`}>
                                                <td className="px-3 py-2">
                                                    <span
                                                        className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                                                        style={{ backgroundColor: (STRATEGY_COLORS[s.strategy] || '#6b7280') + '30', color: STRATEGY_COLORS[s.strategy] || '#9ca3af' }}
                                                    >
                                                        {s.strategy}
                                                    </span>
                                                </td>
                                                <td className="px-3 py-2">{s.n}</td>
                                                <td className="px-3 py-2 text-blue-400">{(s.avg_trust * 100).toFixed(1)}%</td>
                                                <td className="px-3 py-2">{Math.round(s.avg_tokens)}</td>
                                                <td className="px-3 py-2">{Math.round(s.avg_latency)}ms</td>
                                                <td className="px-3 py-2 text-amber-400">{(s.avg_risk * 100).toFixed(1)}%</td>
                                                <td className="px-3 py-2">
                                                    <span className={s.avg_reward >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                                        {s.avg_reward >= 0 ? '+' : ''}{s.avg_reward.toFixed(4)}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ) : (
                        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">No strategy data yet.</div>
                    )}
                </div>
            </div>

            {/* ============ A/B DETAILED COMPARISON ============ */}
            {abTest && !abTest.error && (
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5" id="ab-detail">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">
                        Routing Comparison Detail ({abTest.sample_size} requests)
                    </h3>
                    <div className="grid grid-cols-3 gap-4">
                        {(['always_cot', 'adaptive', 'hybrid'] as const).map(key => {
                            const arm = abTest.arms[key];
                            const isWinner = key === 'adaptive' || (abTest.arms.adaptive.avg_reward < arm.avg_reward);
                            const borderColor = key === 'always_cot' ? 'border-red-500/30' : key === 'adaptive' ? 'border-blue-500/30' : 'border-emerald-500/30';
                            const accentColor = key === 'always_cot' ? 'text-red-400' : key === 'adaptive' ? 'text-blue-400' : 'text-emerald-400';
                            return (
                                <div key={key} className={`border ${borderColor} bg-gray-950 rounded-lg p-4 space-y-3`}>
                                    <div className="flex items-center justify-between">
                                        <h4 className={`text-sm font-bold ${accentColor}`}>{arm.label}</h4>
                                        {key !== 'always_cot' && isWinner && <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">WINNER</span>}
                                    </div>
                                    <div className="space-y-2 text-xs">
                                        <div className="flex justify-between"><span className="text-gray-500">Avg Trust</span><span className="text-gray-200">{(arm.avg_trust * 100).toFixed(1)}%</span></div>
                                        <div className="flex justify-between"><span className="text-gray-500">Avg Tokens</span><span className="text-gray-200">{Math.round(arm.avg_tokens)}</span></div>
                                        <div className="flex justify-between"><span className="text-gray-500">Avg Latency</span><span className="text-gray-200">{Math.round(arm.avg_latency)}ms</span></div>
                                        <div className="flex justify-between"><span className="text-gray-500">Total Tokens</span><span className="text-gray-200">{Math.round(arm.total_tokens).toLocaleString()}</span></div>
                                        <div className="flex justify-between border-t border-gray-800 pt-2 mt-2">
                                            <span className="text-gray-400 font-semibold">Reward</span>
                                            <span className={`font-bold ${arm.avg_reward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {arm.avg_reward >= 0 ? '+' : ''}{arm.avg_reward.toFixed(4)}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
