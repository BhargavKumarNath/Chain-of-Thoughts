import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReasonOps Dashboard",
  description: "Meta-reasoning orchestration layer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen font-mono">
        <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-md sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-8 h-14 flex items-center justify-between">
            <div className="flex items-center gap-6">
              <span className="font-bold text-blue-500 text-lg">ReasonOps</span>
              <div className="flex gap-4 text-sm">
                <Link href="/" className="text-gray-400 hover:text-blue-400 transition-colors">Live Console</Link>
                <Link href="/analytics" className="text-gray-400 hover:text-blue-400 transition-colors">Telemetry Dashboard</Link>
              </div>
            </div>
            <div className="text-xs text-gray-600">v0.5.0</div>
          </div>
        </nav>
        <main>
          {children}
        </main>
      </body>
    </html>
  );
}