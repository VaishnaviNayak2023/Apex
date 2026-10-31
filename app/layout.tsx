import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "F1 Strategy Lab | Apex",
  description: "Explore tyre degradation, lap-time models, and race strategy simulations.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
