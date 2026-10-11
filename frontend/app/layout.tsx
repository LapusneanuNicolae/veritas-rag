import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Veritas RAG",
  description: "Ask questions over your documents with verifiable citations.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
