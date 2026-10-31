import { NextRequest, NextResponse } from "next/server";

const backend = process.env.RACE_API_URL ?? "http://127.0.0.1:8000";
const allowedPaths = new Set([
  "backtests",
  "calendar",
  "evaluations",
  "events",
  "health",
  "picks/scoreboard",
  "predictions",
  "race",
  "registry/sync",
  "strategies",
]);

async function forward(request: NextRequest, path: string[]) {
  const endpoint = path.join("/");
  const isRaceId = (value: string | undefined) => Boolean(value && /^\d{4}_\d{2}$/.test(value));
  const isForecastRoute =
    path[0] === "forecasts" &&
    isRaceId(path[1]) &&
    ((path.length === 2 && request.method === "GET") ||
      (path.length === 3 &&
        request.method === "POST" &&
        (path[2] === "update" || path[2] === "what-if")));
  const isPickRoute =
    path[0] === "picks" &&
    path.length === 2 &&
    request.method === "POST" &&
    isRaceId(path[1]);
  if (!allowedPaths.has(endpoint) && !isForecastRoute && !isPickRoute) {
    return NextResponse.json({ detail: "API endpoint not found." }, { status: 404 });
  }

  const destination = new URL(`/api/${endpoint}${request.nextUrl.search}`, backend);
  try {
    const response = await fetch(destination, {
      method: request.method,
      headers: request.method === "POST" ? { "Content-Type": "application/json" } : undefined,
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(120_000),
    });
    const body = await response.text();
    return new NextResponse(body, {
      status: response.status,
      headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "The race-model API is not responding. From the project folder run `npm run dev:all`, or start the Python API with `python -m uvicorn api.main:app --reload --port 8000`.",
      },
      { status: 503 },
    );
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return forward(request, (await context.params).path);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return forward(request, (await context.params).path);
}
