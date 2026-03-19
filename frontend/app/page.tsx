"use client";

import { useState } from 'react';
import { ReasoningResponse, ReasoningRequest } from '../types/reasoning';

export default function ReasoningConsole() {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ReasoningResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        const requestPayload: ReasoningRequest = { query };

        try {
            const res = await fetch(`${apiBaseUrl}/api/v1/reason`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload),
            });

            if (!res.ok) {
                throw new Error(`API Error: ${res.statusText}`);
            }

            const data: ReasoningResponse = await res.json();
            setResult(data);
        } catch (err: any) {
            setError(err.message || 'An unknown error occurred.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-mono">
            <div className="max-w-6xl mx-auto space-y-6">
                <header className="border-b border-gray-800 pb-4">
                    <h1 className="text-2xl font-bold text-blue-400">ReasonOps Console</h1>
                    <p className="text-gray-400 text-sm">Meta-reasoning orchestration layer</p>
                </header>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Enter a complex query to test the routing engine (e.g., short vs long queries)..."
                        className="w-full bg-gray-900 border border-gray-700 rounded-md p-4 text-gray-100 focus:outline-none focus:border-blue-500 transition-colors h-32"
                    />
                    <button 
                        type="submit" 
                        disabled={loading || !query.trim()}
                        className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-2 rounded-md font-medium transition-colors"
                    >
                        {loading ? 'Evaluating Complexity & Routing...' : 'Generate Validated Response'}
                    </button>
                </form>

                {error && (
                    <div className="bg-red-900/50 border border-red-500 text-red-200 p-4 rounded-md">
                        {error}
                    </div>
                )}

                {result && (
                    <div className="grid grid-cols-4 gap-6">
                        {/* LEFT COLUMN: Telemetry & Strategy Panel */}
                        <div className="col-span-1 space-y-4">
                            
                            {/* Strategy Panel (New in Phase 2) */}
                            <div className="bg-gray-900 border border-blue-900/50 p-4 rounded-md relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                                <h3 className="text-gray-400 text-xs uppercase mb-3">Active Policy</h3>
                                <div className="space-y-3 text-sm">
                                    <div>
                                        <div className="text-xs text-gray-500 mb-1">Selected Strategy</div>
                                        <div className="font-bold text-blue-400 bg-blue-900/20 py-1 px-2 rounded inline-block">
                                            {result.strategy_selected}
                                        </div>
                                    </div>
                                    <div className="pt-2 border-t border-gray-800">
                                        <div className="flex justify-between mb-1">
                                            <span className="text-gray-400">Hallucination Risk:</span>
                                            <span className={result.hallucination_risk < 0.3 ? 'text-green-400' : result.hallucination_risk < 0.7 ? 'text-yellow-400' : 'text-red-400'}>
                                                {(result.hallucination_risk * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        {/* Progress bar visualizer for risk */}
                                        <div className="w-full bg-gray-800 rounded-full h-1.5">
                                            <div className={`h-1.5 rounded-full ${result.hallucination_risk < 0.3 ? 'bg-green-400' : result.hallucination_risk < 0.7 ? 'bg-yellow-400' : 'bg-red-400'}`} style={{ width: `${result.hallucination_risk * 100}%` }}></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gray-900 border border-gray-800 p-4 rounded-md">
                                <h3 className="text-gray-400 text-xs uppercase mb-2">Execution Telemetry</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Latency:</span>
                                        <span>{result.latency_ms} ms</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Tokens Consumed:</span>
                                        <span>{result.tokens_used}</span>
                                    </div>
                                    <div className="flex justify-between pt-2 border-t border-gray-800 mt-2">
                                        <span className="text-gray-400">Trust Score:</span>
                                        <span className={result.trust_score.aggregate_score > 0.8 ? 'text-green-400' : 'text-yellow-400'}>
                                            {(result.trust_score.aggregate_score * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT COLUMN: Output & Trace */}
                        <div className="col-span-3 space-y-4">
                            <div className="bg-gray-900 border border-gray-800 p-5 rounded-md">
                                <h3 className="text-gray-400 text-xs uppercase mb-3">Verified Output</h3>
                                <p className="text-lg leading-relaxed text-gray-200">{result.final_answer}</p>
                            </div>

                            <div className="bg-gray-900 border border-gray-800 p-5 rounded-md">
                                <h3 className="text-gray-400 text-xs uppercase mb-4">Reasoning Trace</h3>
                                <div className="space-y-4">
                                    {result.reasoning_steps.map((step) => (
                                        <div key={step.step_index} className="flex gap-4 border-l-2 border-gray-700 pl-4">
                                            <div className="text-gray-500 shrink-0 mt-0.5">[{step.step_index}]</div>
                                            <div>
                                                <p className="text-gray-300">{step.content}</p>
                                                {step.assumptions && step.assumptions.length > 0 && (
                                                    <div className="mt-2 text-xs text-gray-500 bg-gray-950 p-2 rounded border border-gray-800 inline-block">
                                                        <span className="font-semibold text-gray-400">Assumptions:</span> {step.assumptions.join(', ')}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
