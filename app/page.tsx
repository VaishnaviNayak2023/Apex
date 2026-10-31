"use client";

import {
  Activity, ArrowDownRight, ArrowUpRight, BarChart3, BookOpenCheck,
  CalendarDays, CircleHelp, FileChartColumnIncreasing, Flag, Gauge, Layers3,
  LoaderCircle, Menu, Moon, Play, RotateCcw, Settings2, ShieldCheck, Sun, Timer, Trophy, X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend,
  Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

type Lap = {
  driver: string; lapNumber: number; lapTime: number; compound: string; tyreLife: number;
  trackStatus?: string | null; trafficGapSeconds?: number | null;
  position?: number | null; positionChange?: number | null;
};
type Metric = { MAE: number; RMSE: number; R2: number };
type RaceEvent = { name: string; round: number; date: string; completed: boolean };
type RegistryEvent = {
  raceId: string; year: number; round: number; grandPrix: string; circuit: string;
  date: string; raceStartUtc?: string; startTimeKnown?: boolean; qualifyingStartUtc?: string | null;
  sessions?: { name: string; startUtc: string; completed?: boolean }[];
  dataTier: string; format: string; status: string;
  predictionLockedAt: string | null; notes: string | null;
};
type RegistryCalendar = {
  year: number; currentYear: number; years: number[]; events: RegistryEvent[];
  defaultRaceId: string | null; source: string; lastUpdated: string | null;
  changes?: { raceId: string; grandPrix: string; type: string; oldValue: string | null; newValue: string | null; detectedAt: string }[];
};
type LedgerPrediction = {
  predictionId: string; raceId: string; predictionType: string; version: number;
  lockedAt: string; modelVersion: string; trainingDataCutoff: string | null;
  payload: {
    bestStrategy?: { compounds?: string[]; pitLaps?: number[] };
    predictedTotalTime?: number;
    pitWindow?: { earliest: number | null; latest: number | null };
    confidence?: { gapToSecondSeconds?: number | null };
    topStrategies?: ForecastStrategy[];
    strategyRanking?: ForecastStrategy[];
    stopCountProbabilities?: Record<string, number>;
    scenarios?: Record<string, { bestStrategy: { compounds: string[]; pitLaps: number[] } | null; note: string }>;
    predictedLapTimes?: { lap: number; lapTime: number }[];
    assumptions?: Record<string, unknown>;
    model?: { hash?: string; version?: string };
  }; year: number;
  grandPrix: string; metrics: Record<string, unknown> | null; causeTag: string | null;
};
type ForecastStrategy = {
  compounds: string[]; pitLaps: number[]; totalTime: number; p10: number; p90: number;
  stops?: number; confidence?: string;
};
type ForecastDetail = {
  race: {
    raceId: string; year: number; grandPrix: string; circuit: string; raceStartUtc: string;
    qualifyingStartUtc: string | null; sessions: {
      name: string; startUtc: string; elapsed: boolean; dataStatus: string;
    }[]; status: string; format: string;
  };
  stage: string; forecastError: string | null;
  earlyForecast: LedgerPrediction | null; updatedForecast: LedgerPrediction | null;
  history: LedgerPrediction[]; confidence: string;
  circuitProfile: Record<string, string | number | null>;
  pastEditions: { raceId: string; year: number; grandPrix: string; date: string; winner: string | null; winnerStrategy: { compoundOrder?: string[]; pitLaps?: number[] } | null; actualsAvailable: boolean }[];
  calendarChanges: { raceId: string; grandPrix: string; type: string; oldValue: string | null; newValue: string | null; detectedAt: string }[];
  freshness: Record<string, string | boolean | null>;
  dataUsed: Record<string, unknown>;
  inputAvailability: Record<string, string>;
  scenarioNotes: Record<string, string>;
};
type WhatIfResult = {
  scenario: string; predictedTime: number; modelPickTime: number; deltaToModelPick: number;
  lapTimes: { lap: number; lapTime: number }[]; modelPickLapTimes: { lap: number; lapTime: number }[];
  modelPick: { compounds: string[]; pitLaps: number[] };
  strategy: { compounds: string[]; pitLaps: number[] };
};
type PickScoreboard = {
  evaluated: number; exactMatches: number; hitRate: number | null;
  picks: { raceId: string; strategy: { compounds: string[]; pitLaps: number[] }; createdAt: string; evaluated: boolean; exactMatch: boolean | null }[];
};
type EvaluationSummary = {
  evaluatedRaces: number; stopCountHitRate: number | null;
  meanPitLapError: number | null; meanLapTimeMae: number | null;
};
type EvaluationResponse = { evaluations: LedgerPrediction[]; summary: EvaluationSummary };
type BacktestRun = {
  runId: string; createdAt: string; startYear: number; endYear: number;
  evaluatedRaces: number; stopCountHitRate: number | null;
  meanPitLapError: number | null; meanLapTimeMae: number | null;
};
type ScenarioDefaults = {
  trackTemp: number; airTemp: number; trafficIntensity: number;
  overtakeRate: number; safetyCarProbability: number; modelError: number;
};
type RaceData = {
  year: number; race: string; lapCount: number; driverCount: number; drivers: string[];
  teams: string[]; defaultTeam: string; totalLaps: number; source: string;
  scenarioDefaults: ScenarioDefaults;
  loadWarning?: string; bestModel: string; bestModelMae: number;
  metrics: Record<string, Metric>; laps: Lap[];
  strategies: StrategyRow[];
  degradation: { compound: string; tyreAge: number; lapTime: number }[];
};
type StrategyRow = {
  strategy: string; totalTime: number; delta: number;
  p10: number; p90: number; winProbability: number;
};
type EventsResponse = {
  year: number; currentYear: number; defaultEvent: string;
  years: number[]; events: RaceEvent[];
};
type PageName = "Overview" | "Race Explorer" | "Tyre Degradation" | "Model Performance" | "Strategy Simulator" | "Race Calendar" | "Next Race Prediction" | "Prediction Ledger" | "Race Reports" | "Methodology";

const pages: { name: PageName; icon: typeof Activity }[] = [
  { name: "Overview", icon: Gauge },
  { name: "Race Explorer", icon: Flag },
  { name: "Tyre Degradation", icon: Activity },
  { name: "Model Performance", icon: BarChart3 },
  { name: "Strategy Simulator", icon: Layers3 },
  { name: "Race Calendar", icon: CalendarDays },
  { name: "Next Race Prediction", icon: Flag },
  { name: "Prediction Ledger", icon: BookOpenCheck },
  { name: "Race Reports", icon: FileChartColumnIncreasing },
  { name: "Methodology", icon: CircleHelp },
];
const compoundColors: Record<string, string> = { SOFT: "#ff4d55", MEDIUM: "#ffd166", HARD: "#e7eaf0" };
const fmt = (value: number, digits = 1) => value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{children}</section>;
}
function SectionHeading({ title, note }: { title: string; note: string }) {
  return <div className="section-heading"><div><h2>{title}</h2><p>{note}</p></div><button className="icon-button" aria-label="More information"><CircleHelp size={16} /></button></div>;
}
function CompoundTag({ compound }: { compound: string }) {
  return <span className={`compound-tag ${compound.toLowerCase()}`}><i />{compound}</span>;
}
function strategyLabel(strategy?: { compounds?: string[]; pitLaps?: number[] }) {
  if (!strategy) return "—";
  const compounds = strategy.compounds?.map((compound) => compound[0]).join(" → ") ?? "—";
  const pits = strategy.pitLaps?.length ? ` · pit ${strategy.pitLaps.join(", ")}` : "";
  return `${compounds}${pits}`;
}
function trackStatusLabel(status?: string | null) {
  return ({
    "1": "GREEN",
    "2": "YELLOW",
    "4": "SAFETY CAR",
    "5": "RED FLAG",
    "6": "VIRTUAL SC",
    "8": "CHEQUERED",
  } as Record<string, string>)[status ?? ""] ?? status ?? "—";
}

export default function Dashboard() {
  const [page, setPage] = useState<PageName>("Overview");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [mobileNav, setMobileNav] = useState(false);
  const [race, setRace] = useState<RaceData | null>(null);
  const currentYear = new Date().getFullYear();
  const [selectedYear, setSelectedYear] = useState(currentYear);
  const [selectedRace, setSelectedRace] = useState("");
  const [availableYears, setAvailableYears] = useState<number[]>([]);
  const [events, setEvents] = useState<RaceEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedDriver, setSelectedDriver] = useState("All drivers");
  const [stops, setStops] = useState(1);
  const [raceLaps, setRaceLaps] = useState(57);
  const [pitLoss, setPitLoss] = useState(22);
  const [trackTemp, setTrackTemp] = useState(38);
  const [airTemp, setAirTemp] = useState(27);
  const [weatherChange, setWeatherChange] = useState(0);
  const [trafficIntensity, setTrafficIntensity] = useState(0.15);
  const [overtakeRate, setOvertakeRate] = useState(0.25);
  const [safetyCarProbability, setSafetyCarProbability] = useState(0.2);
  const [safetyCarDuration, setSafetyCarDuration] = useState(4);
  const [safetyCarPitSaving, setSafetyCarPitSaving] = useState(8);
  const [safetyCarLapDelta, setSafetyCarLapDelta] = useState(28);
  const [availableSets, setAvailableSets] = useState({ SOFT: 2, MEDIUM: 3, HARD: 2 });
  const [selectedTeam, setSelectedTeam] = useState("Average");
  const [strategies, setStrategies] = useState<StrategyRow[]>([]);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState("");
  const [registryYear, setRegistryYear] = useState(currentYear);
  const [registry, setRegistry] = useState<RegistryCalendar | null>(null);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [registryError, setRegistryError] = useState("");
  const [ledger, setLedger] = useState<LedgerPrediction[]>([]);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [ledgerError, setLedgerError] = useState("");
  const [nextEvent, setNextEvent] = useState<RegistryEvent | null>(null);
  const [nextPrediction, setNextPrediction] = useState<LedgerPrediction | null>(null);
  const [forecastDetail, setForecastDetail] = useState<ForecastDetail | null>(null);
  const [nextLoading, setNextLoading] = useState(false);
  const [nextError, setNextError] = useState("");
  const [updateForecastLoading, setUpdateForecastLoading] = useState(false);
  const [clock, setClock] = useState(Date.now());
  const [whatIfCompounds, setWhatIfCompounds] = useState("MEDIUM,HARD");
  const [whatIfPitLap, setWhatIfPitLap] = useState("25");
  const [whatIfScenario, setWhatIfScenario] = useState("baseline");
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResult | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState("");
  const [pickScoreboard, setPickScoreboard] = useState<PickScoreboard | null>(null);
  const [pickMessage, setPickMessage] = useState("");
  const [reminderEnabled, setReminderEnabled] = useState(false);
  const [reports, setReports] = useState<EvaluationResponse | null>(null);
  const [backtests, setBacktests] = useState<BacktestRun[] | null>(null);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [reportsError, setReportsError] = useState("");

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("apex-theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      setTheme(savedTheme);
    } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
      setTheme("light");
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    window.localStorage.setItem("apex-theme", nextTheme);
  };

  const loadEvents = useCallback(async (year: number, refresh = false) => {
    setEventsLoading(true);
    setError("");
    try {
      const response = await fetch(`/api/events?year=${year}${refresh ? "&refresh=true" : ""}`, { cache: "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? `Season calendar returned ${response.status}`);
      const schedule = data as EventsResponse;
      setEvents(schedule.events);
      setAvailableYears(schedule.years);
      setSelectedRace(schedule.defaultEvent);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the Formula 1 calendar.");
      setEvents([]);
      setSelectedRace("");
    } finally {
      setEventsLoading(false);
    }
  }, []);

  const loadRace = useCallback(async (refresh = false) => {
    if (!selectedRace) return;
    setLoading(true);
    setError("");
    setRace(null);
    try {
      const query = new URLSearchParams({ year: String(selectedYear), race: selectedRace });
      if (refresh) query.set("refresh", "true");
      const response = await fetch(`/api/race?${query}`, { cache: "no-store" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Race API returned ${response.status}`);
      const data = body as RaceData;
      setRace(data);
      setStrategies(data.strategies);
      setRaceLaps(data.totalLaps);
      setSelectedTeam(data.defaultTeam);
      setTrackTemp(data.scenarioDefaults.trackTemp);
      setAirTemp(data.scenarioDefaults.airTemp);
      setTrafficIntensity(data.scenarioDefaults.trafficIntensity);
      setOvertakeRate(data.scenarioDefaults.overtakeRate);
      setSafetyCarProbability(data.scenarioDefaults.safetyCarProbability);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load race data.");
    } finally {
      setLoading(false);
    }
  }, [selectedRace, selectedYear]);

  const runSimulation = useCallback(async () => {
    if (!selectedRace) return;
    setSimLoading(true);
    setSimError("");
    try {
      const response = await fetch("/api/strategies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          year: selectedYear, race: selectedRace, total_laps: raceLaps, pit_loss: pitLoss,
          stops, track_temp: trackTemp, air_temp: airTemp, weather_change: weatherChange,
          traffic_intensity: trafficIntensity, overtake_rate: overtakeRate,
          safety_car_probability: safetyCarProbability, safety_car_duration: safetyCarDuration,
          safety_car_pit_saving: safetyCarPitSaving, safety_car_lap_delta: safetyCarLapDelta,
          available_sets: availableSets, team: selectedTeam,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? `Simulation API returned ${response.status}`);
      setStrategies(result.strategies as StrategyRow[]);
    } catch (reason) {
      setSimError(reason instanceof Error ? reason.message : "Could not simulate strategies.");
    } finally {
      setSimLoading(false);
    }
  }, [
    selectedRace, selectedYear, raceLaps, pitLoss, stops, trackTemp, airTemp,
    weatherChange, trafficIntensity, overtakeRate, safetyCarProbability,
    safetyCarDuration, safetyCarPitSaving, safetyCarLapDelta, availableSets, selectedTeam,
  ]);

  const loadRegistry = useCallback(async () => {
    setRegistryLoading(true);
    setRegistryError("");
    setRegistry(null);
    try {
      const response = await fetch(`/api/calendar?year=${registryYear}`, { cache: "no-store" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Race registry returned ${response.status}`);
      setRegistry(body as RegistryCalendar);
    } catch (reason) {
      setRegistryError(reason instanceof Error ? reason.message : "Could not load the race registry.");
    } finally {
      setRegistryLoading(false);
    }
  }, [registryYear]);

  const syncRegistry = useCallback(async () => {
    setRegistryLoading(true);
    setRegistryError("");
    try {
      const response = await fetch("/api/registry/sync", { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Registry refresh returned ${response.status}`);
      await loadRegistry();
    } catch (reason) {
      setRegistryError(reason instanceof Error ? reason.message : "Could not refresh the race registry.");
      setRegistryLoading(false);
    }
  }, [loadRegistry]);

  const loadLedger = useCallback(async () => {
    setLedgerLoading(true);
    setLedgerError("");
    setLedger([]);
    try {
      const response = await fetch(`/api/predictions?year=${registryYear}`, { cache: "no-store" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Prediction ledger returned ${response.status}`);
      setLedger(body.predictions as LedgerPrediction[]);
    } catch (reason) {
      setLedgerError(reason instanceof Error ? reason.message : "Could not load the prediction ledger.");
    } finally {
      setLedgerLoading(false);
    }
  }, [registryYear]);

  const loadNextRace = useCallback(async () => {
    setNextLoading(true);
    setNextError("");
    setNextEvent(null);
    setNextPrediction(null);
    setForecastDetail(null);
    setWhatIfResult(null);
    setPickScoreboard(null);
    try {
      const [calendarResponse, predictionResponse] = await Promise.all([
        fetch(`/api/calendar?year=${currentYear}`, { cache: "no-store" }),
        fetch(`/api/predictions?year=${currentYear}`, { cache: "no-store" }),
      ]);
      const calendarBody = await calendarResponse.json();
      const predictionBody = await predictionResponse.json();
      if (!calendarResponse.ok) throw new Error(calendarBody.detail ?? `Calendar returned ${calendarResponse.status}`);
      if (!predictionResponse.ok) throw new Error(predictionBody.detail ?? `Ledger returned ${predictionResponse.status}`);
      const upcoming = (calendarBody as RegistryCalendar).events.find(
        (event) => ["scheduled", "predicted", "early_forecast", "updated_forecast"].includes(event.status),
      ) ?? null;
      setNextEvent(upcoming);
      if (upcoming) {
        const predictions = predictionBody.predictions as LedgerPrediction[];
        const existingPrediction = predictions.find(
          (prediction) => prediction.raceId === upcoming.raceId && prediction.predictionType === "P1",
        ) ?? predictions.find(
          (prediction) => prediction.raceId === upcoming.raceId && prediction.predictionType === "P0",
        ) ?? null;
        const detailResponse = await fetch(`/api/forecasts/${encodeURIComponent(upcoming.raceId)}`, { cache: "no-store" });
        const detailBody = await detailResponse.json();
        if (!detailResponse.ok) throw new Error(detailBody.detail ?? `Forecast returned ${detailResponse.status}`);
        const detail = detailBody as ForecastDetail;
        setForecastDetail(detail);
        setNextPrediction(detail.updatedForecast ?? detail.earlyForecast ?? existingPrediction);
        setWhatIfCompounds((detail.updatedForecast ?? detail.earlyForecast)?.payload.bestStrategy?.compounds?.join(",") ?? "MEDIUM,HARD");
        setWhatIfPitLap(String((detail.updatedForecast ?? detail.earlyForecast)?.payload.bestStrategy?.pitLaps?.[0] ?? 25));
        if (typeof window !== "undefined" && "Notification" in window) {
          setReminderEnabled(localStorage.getItem(`forecast-reminder:${upcoming.raceId}`) === "true");
        }
        const ownerId = localStorage.getItem("f1-strategy-local-owner");
        if (ownerId) {
          const scoreboardResponse = await fetch(`/api/picks/scoreboard?owner_id=${encodeURIComponent(ownerId)}`, { cache: "no-store" });
          const scoreboardBody = await scoreboardResponse.json();
          if (scoreboardResponse.ok) setPickScoreboard(scoreboardBody as PickScoreboard);
          else setPickMessage(scoreboardBody.detail ?? "Could not load your pre-race pick scoreboard.");
        }
      }
    } catch (reason) {
      setNextError(reason instanceof Error ? reason.message : "Could not load the next-race prediction.");
    } finally {
      setNextLoading(false);
    }
  }, [currentYear]);

  const updateForecast = useCallback(async () => {
    if (!nextEvent) return;
    setUpdateForecastLoading(true);
    setNextError("");
    try {
      const response = await fetch(`/api/forecasts/${encodeURIComponent(nextEvent.raceId)}/update`, { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Post-qualifying forecast returned ${response.status}`);
      await loadNextRace();
    } catch (reason) {
      setNextError(reason instanceof Error ? reason.message : "Could not generate the post-qualifying forecast.");
    } finally {
      setUpdateForecastLoading(false);
    }
  }, [loadNextRace, nextEvent]);

  const submitWhatIf = useCallback(async () => {
    if (!nextEvent) return;
    const compounds = whatIfCompounds.split(",").map((value) => value.trim().toUpperCase()).filter(Boolean);
    const pitLaps = whatIfPitLap.split(",").map((value) => Number(value.trim())).filter(Number.isFinite);
    setWhatIfLoading(true);
    setWhatIfError("");
    try {
      const response = await fetch(`/api/forecasts/${encodeURIComponent(nextEvent.raceId)}/what-if`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ compounds, pit_laps: pitLaps, scenario: whatIfScenario }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `What-if simulation returned ${response.status}`);
      setWhatIfResult(body as WhatIfResult);
    } catch (reason) {
      setWhatIfError(reason instanceof Error ? reason.message : "Could not simulate this strategy.");
    } finally {
      setWhatIfLoading(false);
    }
  }, [nextEvent, whatIfCompounds, whatIfPitLap, whatIfScenario]);

  const submitPreRacePick = useCallback(async () => {
    if (!nextEvent) return;
    const compounds = whatIfCompounds.split(",").map((value) => value.trim().toUpperCase()).filter(Boolean);
    const pitLaps = whatIfPitLap.split(",").map((value) => Number(value.trim())).filter(Number.isFinite);
    setPickMessage("");
    try {
      const storageKey = "f1-strategy-local-owner";
      let ownerId = localStorage.getItem(storageKey);
      if (!ownerId) {
        ownerId = crypto.randomUUID();
        localStorage.setItem(storageKey, ownerId);
      }
      const response = await fetch(`/api/picks/${encodeURIComponent(nextEvent.raceId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ owner_id: ownerId, compounds, pit_laps: pitLaps }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Pick returned ${response.status}`);
      setPickMessage(`Pre-race pick saved at ${new Date(body.createdAt).toLocaleString()}.`);
      const scoreboardResponse = await fetch(`/api/picks/scoreboard?owner_id=${encodeURIComponent(ownerId)}`, { cache: "no-store" });
      const scoreboardBody = await scoreboardResponse.json();
      if (!scoreboardResponse.ok) throw new Error(scoreboardBody.detail ?? "Could not refresh the pick scoreboard.");
      setPickScoreboard(scoreboardBody as PickScoreboard);
    } catch (reason) {
      setPickMessage(reason instanceof Error ? reason.message : "Could not save your pre-race pick.");
    }
  }, [nextEvent, whatIfCompounds, whatIfPitLap]);

  const toggleForecastReminder = useCallback(async () => {
    if (!nextEvent || !("Notification" in window)) {
      setPickMessage("Browser notifications are unavailable in this browser.");
      return;
    }
    if (!reminderEnabled) {
      const permission = await Notification.requestPermission();
      if (permission !== "granted") {
        setPickMessage("Notification permission was not granted.");
        return;
      }
    }
    const enabled = !reminderEnabled;
    localStorage.setItem(`forecast-reminder:${nextEvent.raceId}`, String(enabled));
    setReminderEnabled(enabled);
    setPickMessage(enabled ? "Browser reminder enabled while this dashboard is open." : "Browser reminder disabled.");
  }, [nextEvent, reminderEnabled]);

  const downloadForecastCard = useCallback(() => {
    if (!nextEvent || !nextPrediction) return;
    const canvas = document.createElement("canvas");
    canvas.width = 1200;
    canvas.height = 630;
    const context = canvas.getContext("2d");
    if (!context) {
      setPickMessage("This browser could not create the share card.");
      return;
    }
    context.fillStyle = "#101216";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#e10600";
    context.fillRect(72, 72, 8, 486);
    context.fillStyle = "#f0f1f4";
    context.font = "600 28px Arial";
    context.fillText("APEX / STRATEGY · PROVISIONAL FORECAST", 112, 115);
    context.font = "600 54px Arial";
    context.fillText(`${nextEvent.year} ${nextEvent.grandPrix}`, 112, 210, 980);
    context.font = "32px Arial";
    context.fillStyle = "#b8bec9";
    context.fillText(nextEvent.circuit, 112, 265, 980);
    context.fillStyle = "#f0f1f4";
    context.font = "600 38px Arial";
    context.fillText(strategyLabel(nextPrediction.payload.bestStrategy), 112, 365, 980);
    context.font = "26px Arial";
    context.fillStyle = "#b8bec9";
    const pitWindow = nextPrediction.payload.pitWindow;
    context.fillText(`Pit window ${pitWindow?.earliest ?? "—"}–${pitWindow?.latest ?? "—"} · ${forecastDetail?.confidence ?? "Unrated"} confidence`, 112, 425);
    context.fillText(`Forecast recorded ${new Date(nextPrediction.lockedAt).toLocaleString()} · cutoff ${nextPrediction.trainingDataCutoff ?? "unavailable"}`, 112, 490, 980);
    context.font = "20px Arial";
    context.fillStyle = "#858c98";
    context.fillText("Historical-data estimate. Not official team or race-control timing.", 112, 550);
    canvas.toBlob((blob) => {
      if (!blob) {
        setPickMessage("The share card could not be exported.");
        return;
      }
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${nextEvent.raceId}-forecast.png`;
      anchor.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }, [forecastDetail?.confidence, nextEvent, nextPrediction]);

  const loadReports = useCallback(async () => {
    setReportsLoading(true);
    setReportsError("");
    setReports(null);
    setBacktests(null);
    try {
      const [response, backtestResponse] = await Promise.all([
        fetch(`/api/evaluations?year=${registryYear}`, { cache: "no-store" }),
        fetch(`/api/backtests?year=${registryYear}`, { cache: "no-store" }),
      ]);
      const [body, backtestBody] = await Promise.all([response.json(), backtestResponse.json()]);
      if (!response.ok) throw new Error(body.detail ?? `Race reports returned ${response.status}`);
      if (!backtestResponse.ok) throw new Error(backtestBody.detail ?? `Backtest reports returned ${backtestResponse.status}`);
      setReports(body as EvaluationResponse);
      setBacktests(backtestBody.runs as BacktestRun[]);
    } catch (reason) {
      setReportsError(reason instanceof Error ? reason.message : "Could not load race reports.");
    } finally {
      setReportsLoading(false);
    }
  }, [registryYear]);

  useEffect(() => { void loadEvents(selectedYear); }, [loadEvents, selectedYear]);
  useEffect(() => { void loadRace(); }, [loadRace]);
  useEffect(() => {
    if (page === "Race Calendar") void loadRegistry();
    if (page === "Prediction Ledger") void loadLedger();
    if (page === "Next Race Prediction") void loadNextRace();
    if (page === "Race Reports") void loadReports();
  }, [page, loadLedger, loadNextRace, loadRegistry, loadReports]);
  useEffect(() => {
    if (page === "Strategy Simulator" && race) void runSimulation();
  }, [page, race, runSimulation]);
  useEffect(() => {
    const timer = window.setInterval(() => setClock(Date.now()), 30_000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!reminderEnabled || !nextEvent || !forecastDetail || !("Notification" in window)) return;
    const key = `forecast-stage:${nextEvent.raceId}`;
    if (!localStorage.getItem(key)) localStorage.setItem(key, forecastDetail.stage);
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/forecasts/${encodeURIComponent(nextEvent.raceId)}`, { cache: "no-store" });
        if (!response.ok) return;
        const detail = await response.json() as ForecastDetail;
        const previousStage = localStorage.getItem(key);
        if (previousStage && previousStage !== detail.stage && Notification.permission === "granted") {
          new Notification(`${nextEvent.grandPrix} forecast updated`, {
            body: `Forecast stage changed from ${previousStage} to ${detail.stage}.`,
          });
        }
        localStorage.setItem(key, detail.stage);
        setForecastDetail(detail);
      } catch {
        setPickMessage("Could not check for a forecast update; the reminder will retry.");
      }
    }, 5 * 60_000);
    return () => window.clearInterval(timer);
  }, [forecastDetail, nextEvent, reminderEnabled]);

  const shownLaps = useMemo(() => {
    if (!race) return [];
    return selectedDriver === "All drivers" ? race.laps : race.laps.filter((lap) => lap.driver === selectedDriver);
  }, [race, selectedDriver]);
  const best = strategies[0];
  const runnerUp = strategies[1];
  const strategyComparison = useMemo(() => {
    if (!whatIfResult) return [];
    let gap = 0;
    return whatIfResult.lapTimes.map((lap, index) => {
      const modelLap = whatIfResult.modelPickLapTimes[index];
      gap += lap.lapTime - (modelLap?.lapTime ?? lap.lapTime);
      return { lap: lap.lap, cumulativeGap: Number(gap.toFixed(2)) };
    });
  }, [whatIfResult]);
  const countdownMs = forecastDetail
    ? new Date(forecastDetail.race.raceStartUtc).getTime() - clock
    : null;
  const qualifyingReady = Boolean(forecastDetail?.race.qualifyingStartUtc)
    && clock >= new Date(forecastDetail?.race.qualifyingStartUtc ?? 0).getTime() + 2 * 60 * 60 * 1000
    && countdownMs !== null && countdownMs > 0;
  const chartTooltip = theme === "dark"
    ? { contentStyle: { background: "#171b24", border: "1px solid #2a303d", borderRadius: 10, color: "#f3f5f8" }, itemStyle: { color: "#f3f5f8" }, labelStyle: { color: "#9ba3b4" } }
    : { contentStyle: { background: "#ffffff", border: "1px solid #d5dce7", borderRadius: 10, color: "#172033" }, itemStyle: { color: "#172033" }, labelStyle: { color: "#536176" } };

  return (
    <div className="app-shell" data-theme={theme}>
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand"><div className="brand-mark"><Flag size={19} /></div><div><strong>APEX<span> / STRATEGY</span></strong><small>RACE INTELLIGENCE</small></div><button className="close-nav" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={18} /></button></div>
        <div className="race-select-label">EVENT SELECTOR</div>
        <div className="race-selector-fields">
          <label>SEASON<select aria-label="Season" value={selectedYear} onChange={(event) => { setRace(null); setSelectedRace(""); setSelectedYear(Number(event.target.value)); }}>
            {(availableYears.length ? availableYears : [currentYear]).map((year) => <option key={year} value={year}>{year} FIA Formula One Championship</option>)}
          </select></label>
          <label>GRAND PRIX<select aria-label="Grand Prix" value={selectedRace} disabled={eventsLoading || events.length === 0} onChange={(event) => setSelectedRace(event.target.value)}>
            {events.length === 0 && <option value="">{eventsLoading ? "Loading season calendar…" : "Calendar unavailable"}</option>}
            {events.map((event) => <option key={event.round} value={event.name}>{String(event.round).padStart(2, "0")} · {event.name}{event.completed ? "" : " · UPCOMING"}</option>)}
          </select></label>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="nav-list" aria-label="Dashboard sections">
          {pages.map(({ name, icon: Icon }) => <button key={name} onClick={() => { setPage(name); setMobileNav(false); }} className={`nav-item ${page === name ? "active" : ""}`}><Icon size={17} /><span>{name}</span>{name === "Strategy Simulator" && <span className="nav-live-dot" />}</button>)}
        </nav>
        <div className="sidebar-bottom">
          <div className="system-status"><span className="status-dot" /><span><b>MODEL SYSTEMS</b><small>{race?.source === "FastF1" ? "FastF1 timing loaded" : race?.source === "OpenF1" ? "OpenF1 historical timing loaded" : race ? "Demo data clearly marked" : "Waiting for race service"}</small></span><ShieldCheck size={17} /></div>
          <button className="profile"><span className="avatar">AP</span><span><b>Analyst workspace</b><small>Race strategy lab</small></span><Settings2 size={16} /></button>
        </div>
      </aside>
      {mobileNav && <button className="nav-backdrop" onClick={() => setMobileNav(false)} aria-label="Close menu" />}

      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu icon-button" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumbs"><span>WORKSPACE</span><b>/</b><strong>{page.toUpperCase()}</strong></div>
          <div className="topbar-right"><span className="session-badge"><span />{selectedYear} SEASON</span><button className="icon-button theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}<span>{theme === "dark" ? "Light" : "Dark"}</span></button><button className="icon-button refresh-button" onClick={() => { void loadEvents(selectedYear, true); if (selectedRace) void loadRace(true); }} aria-label="Refresh race and calendar data" title="Refresh current calendar and race data"><RotateCcw size={15} /></button></div>
        </header>

        <div className="page-wrap">
          {(loading || eventsLoading) && <div className="notice loading-notice"><LoaderCircle className="spin" size={16} />{eventsLoading ? "Loading the current Formula 1 calendar…" : `Loading ${selectedYear} ${selectedRace} race data and model…`}</div>}
          {error && <div className="notice error-notice"><span>{error}</span><button onClick={() => { void loadEvents(selectedYear, true); if (selectedRace) void loadRace(true); }}>Retry</button></div>}
          {race?.loadWarning && <div className="notice data-notice"><Activity size={16} /><span>{race.loadWarning}{race.source === "Demo data" ? " This is synthetic demonstration data, not official race timing." : ""}</span></div>}

          {page !== "Overview" && <div className="page-title-row">
            <div className="title-kicker"><span className="kicker-line" />{selectedYear} FIA FORMULA ONE WORLD CHAMPIONSHIP</div>
            <h1>{page}</h1>
            <p className="page-subtitle">{page === "Strategy Simulator" ? "Stress-test pit windows and compound choices against predicted full-race pace." : page === "Race Calendar" ? "Explore the generated championship registry from 1950 through the current season." : page === "Next Race Prediction" ? "A provisional historical-data forecast with a visible cutoff, stage, and input provenance." : page === "Prediction Ledger" ? "Review pre-race calls, timestamps, model versions, and post-race scores." : page === "Race Reports" ? "Compare locked predictions to official race outcomes and review season-level scoring." : `${selectedRace || "Select a Grand Prix"} · ${race?.totalLaps ?? "—"} laps · Data-led ${page.toLowerCase()} for race strategy.`}</p>
          </div>}

          {page === "Overview" && <section className="home-hero" aria-labelledby="home-heading">
            <div className="home-hero-copy">
              <div className="home-eyebrow"><span className="home-live-dot" /> APEX RACE INTELLIGENCE <span className="eyebrow-divider">/</span> {selectedYear} SEASON</div>
              <h1 id="home-heading"><span className="home-heading-line">Every second has</span>{" "}<span className="home-heading-line"><i>a strategy.</i></span></h1>
              <p>Explore race pace, tyre life, and pit windows with data-led simulations. Find the move that puts you ahead.</p>
              <div className="home-hero-actions">
                <button className="primary-button" onClick={() => setPage("Strategy Simulator")}><Play size={15} fill="currentColor" />Launch race simulator<ArrowUpRight size={15} /></button>
                <button className="hero-secondary-button" onClick={() => setPage("Race Explorer")}>Explore race data<ArrowUpRight size={14} /></button>
              </div>
              <div className="hero-trust"><ShieldCheck size={15} /><span>Historical timing and explicit simulations</span><i /></div>
            </div>
            <div className="home-hero-visual" aria-label="Illustration of a Formula racing car on a circuit">
              <div className="visual-topline"><span><i /> STRATEGY SIMULATION</span><span>NO. {String((events.find((event) => event.name === selectedRace)?.round ?? 1)).padStart(2, "0")}</span></div>
              <svg className="track-art" viewBox="0 0 640 330" role="img" aria-labelledby="track-title">
                <title id="track-title">Stylized Formula race car and circuit</title>
                <defs>
                  <linearGradient id="car-red" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stopColor="#ff5c55" /><stop offset="1" stopColor="#ad1118" /></linearGradient>
                  <linearGradient id="track-glow" x1="0" x2="1"><stop offset="0" stopColor="#e51b23" stopOpacity=".05" /><stop offset=".55" stopColor="#ff343c" stopOpacity=".8" /><stop offset="1" stopColor="#ff343c" stopOpacity=".08" /></linearGradient>
                  <filter id="car-shadow" x="-30%" y="-40%" width="160%" height="180%"><feGaussianBlur stdDeviation="9" /></filter>
                </defs>
                <path className="track-line track-line-back" d="M66 190 C93 86 165 69 226 91 C287 113 279 156 337 158 C394 160 416 99 480 105 C546 111 579 181 552 226 C522 277 452 281 397 255 C334 225 322 204 265 218 C205 233 149 285 99 256 C64 236 55 215 66 190Z" />
                <path className="track-line track-line-glow" d="M66 190 C93 86 165 69 226 91 C287 113 279 156 337 158 C394 160 416 99 480 105 C546 111 579 181 552 226 C522 277 452 281 397 255 C334 225 322 204 265 218 C205 233 149 285 99 256 C64 236 55 215 66 190Z" />
                <path className="track-line track-line-dash" d="M66 190 C93 86 165 69 226 91 C287 113 279 156 337 158 C394 160 416 99 480 105 C546 111 579 181 552 226 C522 277 452 281 397 255 C334 225 322 204 265 218 C205 233 149 285 99 256 C64 236 55 215 66 190Z" />
                <g className="car-illustration" transform="translate(319 184) rotate(-17)">
                  <ellipse cx="0" cy="40" rx="76" ry="18" fill="#ed2028" opacity=".3" filter="url(#car-shadow)" />
                  <rect x="-47" y="-35" width="23" height="15" rx="5" fill="#11151b" stroke="#626b79" strokeWidth="2" />
                  <rect x="24" y="-35" width="23" height="15" rx="5" fill="#11151b" stroke="#626b79" strokeWidth="2" />
                  <rect x="-51" y="23" width="25" height="16" rx="5" fill="#11151b" stroke="#626b79" strokeWidth="2" />
                  <rect x="26" y="23" width="25" height="16" rx="5" fill="#11151b" stroke="#626b79" strokeWidth="2" />
                  <path d="M-65 3 L-47 -4 L-32 -22 L-11 -28 L0 -47 L12 -27 L31 -21 L47 -3 L65 3 L68 12 L42 15 L29 28 L-29 28 L-42 15 L-68 12Z" fill="url(#car-red)" stroke="#ff7774" strokeWidth="1.5" />
                  <path d="M-15 -24 Q0 -42 15 -24 L11 -11 L-11 -11Z" fill="#252c38" stroke="#ffaaa4" strokeWidth="1.5" />
                  <path d="M-10 -25 Q0 -32 10 -25" fill="none" stroke="#d7e6f4" strokeWidth="2" opacity=".85" />
                  <path d="M-37 4 L-24 0 L-21 7 L-36 11Z M37 4 L24 0 L21 7 L36 11Z" fill="#ffcf64" />
                  <path d="M-64 4 L-83 2 L-82 14 L-64 12 M64 4 L83 2 L82 14 L64 12" fill="#e52b32" stroke="#ff7774" strokeWidth="2" />
                  <path d="M-22 26 L-29 39 L29 39 L22 26Z" fill="#d51d25" />
                  <path d="M-35 42 L35 42 L40 48 L-40 48Z" fill="#ff4a50" />
                  <circle cx="0" cy="5" r="3" fill="#ffe7a0" />
                </g>
                <circle className="track-marker" cx="66" cy="190" r="5" />
                <text x="54" y="177" className="track-label">START</text>
              </svg>
              <div className="visual-telemetry">
                <div><span>SELECTED GRAND PRIX</span><strong>{selectedRace || "Season loading"}</strong></div>
                <div><span>RACE DISTANCE</span><strong>{race ? `${race.totalLaps} laps` : "Ready to simulate"}</strong></div>
                <div className="telemetry-stop-control"><span>STRATEGY PLAN</span><div className="hero-stop-toggle" aria-label="Choose simulated pit stops"><button aria-pressed={stops === 1} className={stops === 1 ? "active" : ""} onClick={() => setStops(1)}>1 stop</button><button aria-pressed={stops === 2} className={stops === 2 ? "active" : ""} onClick={() => setStops(2)}>2 stops</button></div></div>
              </div>
            </div>
          </section>}

          {page === "Race Calendar" && <div className="content-stack registry-view">
            <Card className="ranking-card">
              <div className="registry-toolbar">
                <div><h2>Championship calendar</h2><p>Generated from Jolpica-F1 · race rounds, circuit, weekend format, and lifecycle status</p></div>
                <div className="registry-actions">
                  <label htmlFor="registry-season">SEASON<select id="registry-season" className="scenario-select" value={registryYear} onChange={(event) => setRegistryYear(Number(event.target.value))}>
                    {(registry?.years ?? Array.from({ length: currentYear - 1949 }, (_, index) => currentYear - index)).map((year) => <option key={year} value={year}>{year}</option>)}
                  </select></label>
                  <button className="text-button" onClick={() => void syncRegistry()} disabled={registryLoading}>{registryLoading ? "UPDATING…" : "SYNC CALENDAR"}<RotateCcw size={14} /></button>
                </div>
              </div>
              {registryError && <p className="inline-error">{registryError}</p>}
              {registryLoading && <div className="notice loading-notice"><LoaderCircle className="spin" size={16} />Loading race registry…</div>}
              {registry && <>{Boolean(registry.changes?.length) && <div className="notice data-notice calendar-alert"><CalendarDays size={16} /><span>Calendar updates detected: {registry.changes?.map((change) => `${change.grandPrix} ${change.type}`).join(" · ")}</span></div>}<div className="registry-meta"><span>DATA TIER: T0–T3</span><span>SOURCE: {registry.source}</span><span>UPDATED: {registry.lastUpdated ? new Date(registry.lastUpdated).toLocaleString() : "not synced"}</span></div>
                <div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>ROUND</th><th>GRAND PRIX</th><th>CIRCUIT</th><th>DATE</th><th>DATA</th><th>FORMAT</th><th>STATUS</th></tr></thead><tbody>
                  {registry.events.map((event) => <tr key={event.raceId}><td>{String(event.round).padStart(2, "0")}</td><td>{event.grandPrix}</td><td>{event.circuit}</td><td>{event.date}</td><td>{event.dataTier}</td><td>{event.format}</td><td><span className={`status-pill status-${event.status}`}>{event.status.replaceAll("_", " ").toUpperCase()}</span></td></tr>)}
                </tbody></table>
                  {registry.events.length === 0 && <p className="empty-table">No races are listed for this season.</p>}
                </div>
              </>}
              {registryError && !registry && <div className="empty-state"><CalendarDays size={28} /><h2>Calendar unavailable</h2><p>Check network access to Jolpica-F1 and retry.</p></div>}
            </Card>
          </div>}

          {page === "Next Race Prediction" && <div className="content-stack registry-view">
            {nextError && <p className="inline-error">{nextError}</p>}
            {nextLoading && <div className="notice loading-notice"><LoaderCircle className="spin" size={16} />Loading upcoming race and building a historical-only early forecast when data permits…</div>}
            {!nextLoading && nextEvent && forecastDetail && <>
              <div className="notice provisional-notice"><ShieldCheck size={17} /><span><b>PROVISIONAL FORECAST</b> — this is a model estimate from prior race data, not official team strategy or live timing. Target-event data is excluded; see the training cutoff below.</span><button className="text-button" onClick={() => void loadNextRace()} disabled={nextLoading}>REFRESH<RotateCcw size={14} /></button></div>
              {forecastDetail.calendarChanges.length > 0 && <div className="notice data-notice calendar-alert"><CalendarDays size={16} /><span>Calendar change: {forecastDetail.calendarChanges.map((change) => `${change.type} detected ${new Date(change.detectedAt).toLocaleString()}`).join(" · ")}</span></div>}
              <Card className="ranking-card forecast-hero">
                <div className="forecast-hero-top">
                  <div><div className="title-kicker"><span className="kicker-line" />NEXT RACE · {nextEvent.format.toUpperCase()} WEEKEND</div><h2>{nextEvent.year} {nextEvent.grandPrix}</h2><p>{nextEvent.circuit} · {new Date(forecastDetail.race.raceStartUtc).toLocaleString(undefined, { dateStyle: "full", timeStyle: "short" })} ({Intl.DateTimeFormat().resolvedOptions().timeZone})</p></div>
                  <div className="forecast-hero-actions"><span className={`status-pill status-${forecastDetail.stage.toLowerCase().replaceAll(" ", "-")}`}>{forecastDetail.stage.toUpperCase()}</span>
                    {qualifyingReady && !forecastDetail.updatedForecast && <button className="text-button" onClick={() => void updateForecast()} disabled={updateForecastLoading}>{updateForecastLoading ? "UPDATING…" : "PUBLISH POST-QUALIFYING UPDATE"}<RotateCcw size={14} /></button>}
                  </div>
                </div>
                <div className="countdown-block"><span>LIGHTS OUT IN</span><strong>{countdownMs !== null && countdownMs > 0 ? `${Math.floor(countdownMs / 86_400_000)}d ${String(Math.floor((countdownMs % 86_400_000) / 3_600_000)).padStart(2, "0")}h ${String(Math.floor((countdownMs % 3_600_000) / 60_000)).padStart(2, "0")}m` : "Race started or schedule unavailable"}</strong><span>Current stage: {forecastDetail.stage} · {forecastDetail.freshness.forecastGeneratedAt ? `forecast ${new Date(String(forecastDetail.freshness.forecastGeneratedAt)).toLocaleString()}` : "forecast not yet available"}</span></div>
                <div className="weekend-timeline">{forecastDetail.race.sessions.map((session) => <div key={`${session.name}-${session.startUtc}`} className={`timeline-step ${session.elapsed ? "elapsed" : ""}`}><span className="timeline-marker">{session.elapsed ? "✓" : "○"}</span><b>{session.name}</b><small>{new Date(session.startUtc).toLocaleString(undefined, { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</small><small>{session.dataStatus}</small></div>)}</div>
              </Card>
              {forecastDetail.forecastError && <div className="notice data-notice"><Activity size={16} /><span>Forecast unavailable: {forecastDetail.forecastError} The calendar and session preview remain available, but no synthetic substitute is used.</span></div>}
              {nextPrediction ? <>
                <div className="prediction-summary">
                  <div><span>MODEL PICK · {nextPrediction.predictionType}</span><strong>{strategyLabel(nextPrediction.payload.bestStrategy)}</strong></div>
                  <div><span>EXPECTED RACE TIME</span><strong>{nextPrediction.payload.predictedTotalTime ? `${fmt(nextPrediction.payload.predictedTotalTime)} s` : "—"}</strong></div>
                  <div><span>PIT-WINDOW RANGE</span><strong>{nextPrediction.payload.pitWindow?.earliest ?? "—"}–{nextPrediction.payload.pitWindow?.latest ?? "—"}</strong></div>
                </div>
                <div className="forecast-grid">
                  <Card className="ranking-card"><SectionHeading title="Stop-count probabilities" note="Relative model ranking, not calibrated race odds" />
                    <div className="probability-list">{Object.entries(nextPrediction.payload.stopCountProbabilities ?? {}).map(([count, probability]) => <div className="probability-row" key={count}><span>{count}-stop</span><div className="probability-track"><i style={{ width: `${Math.min(100, probability * 100)}%` }} /></div><b>{fmt(probability * 100, 0)}%</b></div>)}</div>
                    <p className="chart-caption">Confidence: {forecastDetail.confidence}. Based on {String(forecastDetail.dataUsed.trainingRaceCount ?? 0)} training races; this is a data-volume label, not a calibrated probability of being correct.</p>
                  </Card>
                  <Card className="ranking-card"><SectionHeading title="Scenario stress tests" note="Model sensitivity only; not a prediction of conditions" />
                    <div className="scenario-cards">{Object.entries(nextPrediction.payload.scenarios ?? {}).map(([name, scenario]) => <div className="scenario-card" key={name}><b>{name === "safetyCar" ? "Safety car" : name === "coolerTrack" ? "Track −5°C" : "Rain"}</b><strong>{scenario.bestStrategy ? strategyLabel(scenario.bestStrategy) : "Not modelled"}</strong><small>{scenario.note}</small></div>)}</div>
                  </Card>
                </div>
                <Card className="ranking-card"><SectionHeading title="Top compound-order options" note="Predicted time and model uncertainty interval" />
                  <div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>RANK</th><th>COMPOUNDS / STOPS</th><th>PIT LAPS</th><th>EXPECTED TIME</th><th>P10–P90</th><th>CONFIDENCE</th></tr></thead><tbody>
                    {(nextPrediction.payload.topStrategies ?? []).slice(0, 3).map((row, index) => <tr key={`${row.compounds.join("-")}-${row.pitLaps.join("-")}`}><td>{String(index + 1).padStart(2, "0")}</td><td>{row.compounds.join(" → ")}</td><td>{row.pitLaps.join(", ") || "No stop"}</td><td>{fmt(row.totalTime)}s</td><td>{fmt(row.p10)}–{fmt(row.p90)}s</td><td>{row.confidence ?? forecastDetail.confidence}</td></tr>)}
                  </tbody></table></div>
                  <div className="registry-meta"><span>LOCKED: {new Date(nextPrediction.lockedAt).toLocaleString()}</span><span>MODEL HASH: {nextPrediction.payload.model?.hash ?? nextPrediction.modelVersion}</span><span>TRAINING CUTOFF: {nextPrediction.trainingDataCutoff ?? "—"}</span></div>
                </Card>
                <div className="forecast-grid">
                  <Card className="ranking-card"><SectionHeading title="Forecast evolution" note="Early forecast compared with post-qualifying update" />
                    {forecastDetail.earlyForecast && forecastDetail.updatedForecast ? <div className="forecast-diff"><p><b>Early:</b> {strategyLabel(forecastDetail.earlyForecast.payload.bestStrategy)}</p><p><b>Post-qualifying time-gated run:</b> {strategyLabel(forecastDetail.updatedForecast.payload.bestStrategy)}</p><p>{strategyLabel(forecastDetail.earlyForecast.payload.bestStrategy) === strategyLabel(forecastDetail.updatedForecast.payload.bestStrategy) ? "The leading compound order is unchanged." : "The leading compound order changed between forecast versions."} The current model does not ingest practice long runs or the qualifying grid; therefore this difference is not attributed to those sessions.</p></div> : <p className="empty-table">The post-qualifying forecast comparison will appear after the updated forecast is locked.</p>}
                    {forecastDetail.history.map((item) => <div className="registry-meta" key={item.predictionId}><span>{item.predictionType} v{item.version}</span><span>{new Date(item.lockedAt).toLocaleString()}</span><span>cutoff {item.trainingDataCutoff ?? "—"}</span></div>)}
                  </Card>
                  <Card className="ranking-card"><SectionHeading title="Circuit preview" note="Known reference details and historical race editions" />
                    <div className="circuit-facts"><span><b>Altitude</b>{forecastDetail.circuitProfile.altitudeMeters ? `${forecastDetail.circuitProfile.altitudeMeters} m` : "Not available"}</span><span><b>Lap length</b>{forecastDetail.circuitProfile.lapLengthKm ? `${forecastDetail.circuitProfile.lapLengthKm} km` : "Not available"}</span><span><b>Tyre wear</b>{String(forecastDetail.circuitProfile.tyreWear ?? "Not available")}</span><span><b>Pit-lane loss</b>{forecastDetail.circuitProfile.pitLaneLossSeconds ? `${forecastDetail.circuitProfile.pitLaneLossSeconds} s` : "Using model assumption if required"}</span><span><b>Typical stops</b>{String(forecastDetail.circuitProfile.typicalStops ?? "Not available")}</span></div>
                    {forecastDetail.circuitProfile.mexicoContext && <p className="chart-caption">{String(forecastDetail.circuitProfile.mexicoContext)}</p>}
                    <p className="chart-caption">Closest-match circuit ranking is unavailable until a comparable-circuit reference set is curated; no matches are inferred from incomplete metadata.</p>
                    <div className="past-editions">{forecastDetail.pastEditions.map((edition) => <div key={edition.raceId}><b>{edition.year}</b><span>{edition.winner ?? "Winner unavailable"}</span><small>{edition.winnerStrategy ? strategyLabel({ compounds: edition.winnerStrategy.compoundOrder, pitLaps: edition.winnerStrategy.pitLaps }) : "Strategy actuals unavailable"}</small></div>)}</div>
                  </Card>
                </div>
                <Card className="ranking-card"><SectionHeading title="Weekend inputs" note="Provider availability is explicit; missing data is not filled with demo timing" />
                  <div className="input-status-grid">{Object.entries(forecastDetail.inputAvailability).map(([name, state]) => <div key={name}><b>{name.replaceAll(/([A-Z])/g, " $1")}</b><span>{state}</span></div>)}</div>
                  <div className="registry-meta"><span>DATA FRESHNESS: {forecastDetail.freshness.calendarLastSynced ? new Date(String(forecastDetail.freshness.calendarLastSynced)).toLocaleString() : "calendar not synced"}</span><span>FASTF1 CREDENTIALS: {forecastDetail.freshness.fastf1CredentialsConfigured ? "configured" : "not configured (optional)"}</span><span>OPENF1: {String(forecastDetail.freshness.openf1Historical)}</span></div>
                </Card>
                <Card className="ranking-card"><SectionHeading title="What-if strategy builder and pre-race pick" note="Compare your dry strategy with the same model and scenario assumptions" />
                  <div className="what-if-controls">
                    <label>COMPOUND ORDER<input value={whatIfCompounds} onChange={(event) => setWhatIfCompounds(event.target.value)} placeholder="MEDIUM,HARD" aria-label="Strategy compound order" /></label>
                    <label>PIT LAP(S)<input value={whatIfPitLap} onChange={(event) => setWhatIfPitLap(event.target.value)} placeholder="25" aria-label="Strategy pit laps" /></label>
                    <label>SCENARIO<select value={whatIfScenario} onChange={(event) => setWhatIfScenario(event.target.value)}><option value="baseline">Baseline</option><option value="safetyCar">Safety-car stress test</option><option value="coolerTrack">Track −5°C stress test</option></select></label>
                    <button className="text-button" onClick={() => void submitWhatIf()} disabled={whatIfLoading}>{whatIfLoading ? "SIMULATING…" : "SIMULATE"}<Play size={14} /></button>
                    <button className="text-button" onClick={() => void submitPreRacePick()} disabled={countdownMs !== null && countdownMs <= 0}>SAVE PRE-RACE PICK<Trophy size={14} /></button>
                  </div>
                  <p className="chart-caption">Enter compounds as comma-separated values (for example MEDIUM,HARD) and pit laps as comma-separated lap numbers. Picks are stored locally under a random browser identifier; this is not account authentication.</p>
                  {whatIfError && <p className="inline-error">{whatIfError}</p>}
                  {pickMessage && <p className="chart-caption">{pickMessage}</p>}
                  {whatIfResult && <div className="what-if-results"><div className="prediction-summary"><div><span>YOUR STRATEGY</span><strong>{fmt(whatIfResult.predictedTime)} s</strong></div><div><span>MODEL PICK</span><strong>{fmt(whatIfResult.modelPickTime)} s</strong></div><div><span>DELTA TO MODEL PICK</span><strong>{whatIfResult.deltaToModelPick > 0 ? "+" : ""}{fmt(whatIfResult.deltaToModelPick)} s</strong></div></div>
                    <div className="comparison-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={strategyComparison}><CartesianGrid stroke="#272d38" vertical={false} /><XAxis dataKey="lap" tick={{ fill: "#858d9d", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#858d9d", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip {...chartTooltip} labelFormatter={(value) => `Lap ${value}`} /><Line dataKey="cumulativeGap" name="Cumulative gap (s)" stroke="#ff6568" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div>
                  </div>}
                  {pickScoreboard && <div className="registry-meta"><span>PICKS: {pickScoreboard.picks.length}</span><span>SCORED: {pickScoreboard.evaluated}</span><span>EXACT MATCHES: {pickScoreboard.exactMatches}</span><span>HIT RATE: {pickScoreboard.hitRate === null ? "—" : `${fmt(pickScoreboard.hitRate * 100, 0)}%`}</span></div>}
                  <div className="share-actions"><button className="text-button" onClick={downloadForecastCard} disabled={!nextPrediction}>DOWNLOAD FORECAST CARD<ArrowDownRight size={14} /></button><button className="text-button" onClick={() => void toggleForecastReminder()}>{reminderEnabled ? "DISABLE BROWSER REMINDER" : "ENABLE BROWSER REMINDER"}<Activity size={14} /></button></div>
                  <p className="chart-caption">Browser reminders work only while this dashboard is open. Email and Telegram delivery require a configured notification provider and are not enabled.</p>
                </Card>
                <details className="data-used-drawer"><summary>Data used, fallbacks and model provenance</summary><pre>{JSON.stringify({ dataUsed: forecastDetail.dataUsed, confidence: forecastDetail.confidence, model: nextPrediction.payload.model, freshness: forecastDetail.freshness, rain: forecastDetail.scenarioNotes.rain }, null, 2)}</pre></details>
              </> : <Card className="ranking-card prediction-card"><div className="empty-state"><Flag size={28} /><h2>No historical-only forecast is available</h2><p>{forecastDetail.forecastError ?? "The race calendar is available, but the model needs verified historical timing before it can produce a forecast. No synthetic race timing is substituted."}</p></div></Card>}
            </>}
            {!nextLoading && !nextError && !nextEvent && <div className="empty-state"><CalendarDays size={28} /><h2>No upcoming race is scheduled</h2><p>Refresh the registry to pick up calendar changes.</p><button className="text-button" onClick={() => void syncRegistry()}>SYNC CALENDAR<RotateCcw size={14} /></button></div>}
          </div>}

          {page === "Prediction Ledger" && <div className="content-stack registry-view">
            <Card className="ranking-card">
              <div className="registry-toolbar">
                <div><h2>Locked prediction ledger</h2><p>Append-only P1 calls with their model version and training cutoff</p></div>
                <div className="registry-actions">
                  <label htmlFor="ledger-season">SEASON<select id="ledger-season" className="scenario-select" value={registryYear} onChange={(event) => setRegistryYear(Number(event.target.value))}>
                    {(registry?.years ?? Array.from({ length: currentYear - 1949 }, (_, index) => currentYear - index)).map((year) => <option key={year} value={year}>{year}</option>)}
                  </select></label>
                  <button className="text-button" onClick={() => void loadLedger()} disabled={ledgerLoading}>{ledgerLoading ? "LOADING…" : "REFRESH LEDGER"}<RotateCcw size={14} /></button>
                </div>
              </div>
              {ledgerError && <p className="inline-error">{ledgerError}</p>}
              {ledgerLoading && <div className="notice loading-notice"><LoaderCircle className="spin" size={16} />Loading prediction ledger…</div>}
              {!ledgerLoading && ledger.length > 0 && <div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>RACE</th><th>TYPE</th><th>LOCKED AT (UTC)</th><th>BEST STRATEGY</th><th>MODEL</th><th>RESULT</th></tr></thead><tbody>
                {ledger.map((prediction) => <tr key={prediction.predictionId}><td>{prediction.year} {prediction.grandPrix}</td><td>{prediction.predictionType} · v{prediction.version}</td><td>{prediction.lockedAt}</td><td>{strategyLabel(prediction.payload.bestStrategy)}</td><td>{prediction.modelVersion}</td><td>{prediction.causeTag ? `${prediction.causeTag} · scored` : "PENDING"}</td></tr>)}
              </tbody></table></div>}
              {!ledgerLoading && !ledgerError && ledger.length === 0 && <div className="empty-state"><BookOpenCheck size={28} /><h2>No locked predictions yet</h2><p>The ledger will only contain calls created before race start from a model trained on earlier, real race data.</p></div>}
            </Card>
          </div>}

          {page === "Race Reports" && <div className="content-stack registry-view">
            <Card className="ranking-card">
              <div className="registry-toolbar">
                <div><h2>Post-race scorecards</h2><p>Scored only against ingested official actuals; incomplete data is not treated as a hit</p></div>
                <div className="registry-actions">
                  <label htmlFor="reports-season">SEASON<select id="reports-season" className="scenario-select" value={registryYear} onChange={(event) => setRegistryYear(Number(event.target.value))}>
                    {(registry?.years ?? Array.from({ length: currentYear - 1949 }, (_, index) => currentYear - index)).map((year) => <option key={year} value={year}>{year}</option>)}
                  </select></label>
                  <button className="text-button" onClick={() => void loadReports()} disabled={reportsLoading}>{reportsLoading ? "LOADING…" : "REFRESH REPORTS"}<RotateCcw size={14} /></button>
                </div>
              </div>
              {reportsError && <p className="inline-error">{reportsError}</p>}
              {reportsLoading && <div className="notice loading-notice"><LoaderCircle className="spin" size={16} />Loading race scorecards…</div>}
              {reports && <><div className="kpi-grid report-kpis">
                <Card className="kpi-card"><div className="kpi-top"><span>EVALUATED RACES</span><FileChartColumnIncreasing size={17} /></div><div className="kpi-value">{reports.summary.evaluatedRaces}</div><div className="kpi-foot">Official actuals only</div></Card>
                <Card className="kpi-card"><div className="kpi-top"><span>STOP-COUNT HIT RATE</span><Trophy size={17} /></div><div className="kpi-value">{reports.summary.stopCountHitRate === null ? "—" : `${fmt(reports.summary.stopCountHitRate * 100, 0)}%`}</div><div className="kpi-foot">Scored P1 predictions</div></Card>
                <Card className="kpi-card"><div className="kpi-top"><span>MEAN PIT-LAP ERROR</span><Timer size={17} /></div><div className="kpi-value">{reports.summary.meanPitLapError === null ? "—" : `${fmt(reports.summary.meanPitLapError, 1)} laps`}</div><div className="kpi-foot">Absolute first-stop error</div></Card>
                <Card className="kpi-card"><div className="kpi-top"><span>MEAN LAP-TIME MAE</span><Gauge size={17} /></div><div className="kpi-value">{reports.summary.meanLapTimeMae === null ? "—" : `${fmt(reports.summary.meanLapTimeMae, 2)} s`}</div><div className="kpi-foot">Comparable winner laps</div></Card>
              </div><div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>RACE</th><th>PREDICTED</th><th>STOP MATCH</th><th>PIT ERROR</th><th>LAP MAE</th><th>ACTUAL RANK</th><th>CAUSE</th></tr></thead><tbody>
                {reports.evaluations.map((report) => <tr key={report.predictionId}><td>{report.year} {report.grandPrix}</td><td>{strategyLabel(report.payload.bestStrategy)}</td><td>{report.metrics?.stopCountMatch === null ? "—" : report.metrics?.stopCountMatch ? "YES" : "NO"}</td><td>{typeof report.metrics?.pitLapError === "number" ? `${report.metrics.pitLapError} laps` : "—"}</td><td>{typeof report.metrics?.lapTimeMae === "number" ? `${fmt(report.metrics.lapTimeMae, 2)} s` : "—"}</td><td>{typeof report.metrics?.actualStrategyRank === "number" ? `#${report.metrics.actualStrategyRank}` : "—"}</td><td>{report.causeTag ?? "—"}</td></tr>)}
              </tbody></table>{reports.evaluations.length === 0 && <p className="empty-table">No official post-race evaluations are available for this season.</p>}</div></>}
            </Card>
            <Card className="ranking-card">
              <div className="registry-toolbar">
                <div><h2>Walk-forward backtests</h2><p>Each race is evaluated using only information available before its start</p></div>
              </div>
              {reportsLoading && <div className="notice loading-notice"><LoaderCircle className="spin" size={16} />Loading backtest results…</div>}
              {backtests && <div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>RUN</th><th>RACE YEARS</th><th>RACES</th><th>STOP HIT RATE</th><th>MEAN PIT ERROR</th><th>MEAN LAP MAE</th></tr></thead><tbody>
                {backtests.map((run) => <tr key={run.runId}><td>{run.runId}</td><td>{run.startYear}–{run.endYear}</td><td>{run.evaluatedRaces}</td><td>{run.stopCountHitRate === null ? "—" : `${fmt(run.stopCountHitRate * 100, 0)}%`}</td><td>{run.meanPitLapError === null ? "—" : `${fmt(run.meanPitLapError, 1)} laps`}</td><td>{run.meanLapTimeMae === null ? "—" : `${fmt(run.meanLapTimeMae, 2)} s`}</td></tr>)}
              </tbody></table>{backtests.length === 0 && <p className="empty-table">No walk-forward backtests are stored for this season yet. Backtests require verified historical timing data.</p>}</div>}
            </Card>
          </div>}

          {race && <div className="content-stack">
            {(page === "Overview" || page === "Strategy Simulator") && <>
              <div className="kpi-grid">
                <Card className="kpi-card"><div className="kpi-top"><span>BEST SIMULATED STRATEGY</span><Trophy size={17} /></div><div className="kpi-value compact">{best?.strategy ?? "—"}</div><div className="kpi-foot"><span className="green-text">OPTIMIZED</span><span>{best ? `${fmt(best.totalTime)}s race time` : "Configure and run a simulation"}</span></div><div className="kpi-accent" /></Card>
                <Card className="kpi-card"><div className="kpi-top"><span>MODEL MAE</span><Gauge size={17} /></div><div className="kpi-value">{race.bestModelMae.toFixed(2)}<small>s</small></div><div className="kpi-foot"><span className="green-text">VALIDATED</span><span>{race.bestModel} · driver holdout</span></div><div className="kpi-accent muted-accent" /></Card>
                <Card className="kpi-card"><div className="kpi-top"><span>LAPS ANALYZED</span><Activity size={17} /></div><div className="kpi-value">{race.lapCount.toLocaleString()}</div><div className="kpi-foot"><span className="blue-text">{race.driverCount} DRIVERS</span><span>Filtered race dataset</span></div><div className="kpi-accent blue-accent" /></Card>
                <Card className="kpi-card"><div className="kpi-top"><span>GAP TO NEXT BEST</span><Timer size={17} /></div><div className="kpi-value">{best && runnerUp ? fmt(runnerUp.totalTime - best.totalTime, 2) : "—"}<small>s</small></div><div className="kpi-foot"><span className="green-text">PREDICTED</span><span>Modelled strategy delta</span></div><div className="kpi-accent gold-accent" /></Card>
              </div>
              <div className="dashboard-grid">
                <Card className="chart-card pace-card"><SectionHeading title="Tyre pace degradation" note="Average lap time by tyre age · all drivers" /><div className="chart-legend"><span><i className="legend-dot soft-dot" />Soft</span><span><i className="legend-dot medium-dot" />Medium</span><span><i className="legend-dot hard-dot" />Hard</span><span className="legend-unit">LAP TIME (S)</span></div>
                  <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={race.degradation} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}><CartesianGrid stroke="#272d38" vertical={false} /><XAxis dataKey="tyreAge" type="number" domain={[1, "dataMax"]} tick={{ fill: "#858d9d", fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#858d9d", fontSize: 11 }} domain={["dataMin - 1", "dataMax + 1"]} axisLine={false} tickLine={false} /><Tooltip {...chartTooltip} labelFormatter={(v) => `Tyre age ${v} laps`} /><Line data={race.degradation.filter((v) => v.compound === "SOFT")} dataKey="lapTime" name="Soft" stroke={compoundColors.SOFT} strokeWidth={2.2} dot={false} activeDot={{ r: 4 }} connectNulls /><Line data={race.degradation.filter((v) => v.compound === "MEDIUM")} dataKey="lapTime" name="Medium" stroke={compoundColors.MEDIUM} strokeWidth={2.2} dot={false} activeDot={{ r: 4 }} connectNulls /><Line data={race.degradation.filter((v) => v.compound === "HARD")} dataKey="lapTime" name="Hard" stroke={compoundColors.HARD} strokeWidth={2.2} dot={false} activeDot={{ r: 4 }} connectNulls /></LineChart></ResponsiveContainer></div><p className="chart-caption">Tyre wear is visible across compounds; fuel-load effects remain in the lap-time signal.</p>
                </Card>
                <Card className="strategy-card"><SectionHeading title="Strategy simulator" note="Adjust race conditions and tyre inventory" />
                  <div className="strategy-input"><label>TEAM PACE MODEL</label><select className="scenario-select" value={selectedTeam} onChange={(event) => setSelectedTeam(event.target.value)}>{race.teams.map((team) => <option key={team}>{team}</option>)}</select></div>
                  <div className="strategy-input"><label>NUMBER OF STOPS</label><div className="segmented-control"><button className={stops === 1 ? "selected" : ""} onClick={() => setStops(1)}>1 STOP</button><button className={stops === 2 ? "selected" : ""} onClick={() => setStops(2)}>2 STOPS</button></div></div>
                  <div className="strategy-input"><label htmlFor="race-laps">RACE DISTANCE <strong>{raceLaps} LAPS</strong></label><input id="race-laps" type="range" min="20" max={Math.max(20, race.totalLaps)} value={Math.min(raceLaps, race.totalLaps)} onChange={(event) => setRaceLaps(Number(event.target.value))} /></div>
                  <div className="strategy-input"><label htmlFor="pit-loss">GREEN-FLAG PIT LOSS <strong>{pitLoss} SEC</strong></label><input id="pit-loss" type="range" min="15" max="35" step="0.5" value={pitLoss} onChange={(event) => setPitLoss(Number(event.target.value))} /></div>
                  <div className="strategy-input"><label htmlFor="track-temp">TRACK TEMPERATURE <strong>{trackTemp.toFixed(1)}°C</strong></label><input id="track-temp" type="range" min="15" max="60" step="0.5" value={trackTemp} onChange={(event) => setTrackTemp(Number(event.target.value))} /></div>
                  <div className="strategy-input"><label htmlFor="air-temp">AIR TEMPERATURE <strong>{airTemp.toFixed(1)}°C</strong></label><input id="air-temp" type="range" min="-5" max="45" step="0.5" value={airTemp} onChange={(event) => setAirTemp(Number(event.target.value))} /></div>
                  <details className="scenario-details"><summary><Settings2 size={14} />TRAFFIC, WEATHER & INCIDENT SCENARIOS</summary>
                    <div className="strategy-input"><label htmlFor="weather-change">WEATHER TREND <strong>{weatherChange > 0 ? "+" : ""}{weatherChange}°C</strong></label><input id="weather-change" type="range" min="-10" max="10" step="0.5" value={weatherChange} onChange={(event) => setWeatherChange(Number(event.target.value))} /><small>Start-to-finish track-temperature change</small></div>
                    <div className="strategy-input"><label htmlFor="traffic-intensity">TRAFFIC EXPOSURE <strong>{Math.round(trafficIntensity * 100)}%</strong></label><input id="traffic-intensity" type="range" min="0" max="1" step="0.05" value={trafficIntensity} onChange={(event) => setTrafficIntensity(Number(event.target.value))} /></div>
                    <div className="strategy-input"><label htmlFor="overtake-rate">OVERTAKE SUCCESS <strong>{Math.round(overtakeRate * 100)}%</strong></label><input id="overtake-rate" type="range" min="0" max="1" step="0.05" value={overtakeRate} onChange={(event) => setOvertakeRate(Number(event.target.value))} /></div>
                    <div className="strategy-input"><label htmlFor="sc-probability">SAFETY-CAR CHANCE <strong>{Math.round(safetyCarProbability * 100)}%</strong></label><input id="sc-probability" type="range" min="0" max="0.8" step="0.05" value={safetyCarProbability} onChange={(event) => setSafetyCarProbability(Number(event.target.value))} /><small>Chance of at least one safety car during the race</small></div>
                    <div className="strategy-input"><label htmlFor="sc-duration">SAFETY-CAR DURATION <strong>{safetyCarDuration} LAPS</strong></label><input id="sc-duration" type="range" min="1" max="10" value={safetyCarDuration} onChange={(event) => setSafetyCarDuration(Number(event.target.value))} /></div>
                    <div className="strategy-input"><label htmlFor="sc-pace">SAFETY-CAR LAP DELTA <strong>+{safetyCarLapDelta} SEC</strong></label><input id="sc-pace" type="range" min="10" max="50" value={safetyCarLapDelta} onChange={(event) => setSafetyCarLapDelta(Number(event.target.value))} /></div>
                    <div className="strategy-input"><label htmlFor="sc-pit-saving">PIT LOSS SAVING UNDER SC <strong>{safetyCarPitSaving} SEC</strong></label><input id="sc-pit-saving" type="range" min="0" max="20" value={safetyCarPitSaving} onChange={(event) => setSafetyCarPitSaving(Number(event.target.value))} /></div>
                    <div className="sets-grid"><span>AVAILABLE TYRE SETS</span>{(["SOFT", "MEDIUM", "HARD"] as const).map((compound) => <label key={compound}><CompoundTag compound={compound} /><input type="number" aria-label={`${compound} tyre sets`} min="0" max="10" value={availableSets[compound]} onChange={(event) => setAvailableSets({ ...availableSets, [compound]: Number(event.target.value) })} /></label>)}</div>
                    <p className="scenario-footnote">Weather drift and traffic/overtaking effects use fitted model responses. Safety-car timing, tyre inventory and model error are sampled in the Monte Carlo scenarios.</p>
                  </details>
                  <button className="simulate-button" onClick={() => void runSimulation()} disabled={simLoading}>{simLoading ? <LoaderCircle size={16} className="spin" /> : <Play size={15} fill="currentColor" />}{simLoading ? "Simulating scenarios…" : "Run scenario simulation"}<ArrowUpRight size={15} /></button>
                  {simError && <p className="inline-error">{simError}</p>}
                </Card>
              </div>
              <Card className="ranking-card"><div className="ranking-header"><SectionHeading title="Top strategy outcomes" note="Monte Carlo expected time, uncertainty and win frequency" /><button className="text-button" onClick={() => setPage("Strategy Simulator")}>VIEW SIMULATOR<ArrowUpRight size={14} /></button></div>
                <div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>RANK</th><th>STRATEGY</th><th>EXPECTED TIME</th><th>P10–P90 RANGE</th><th>WIN CHANCE</th><th>STATUS</th></tr></thead><tbody>{strategies.slice(0, 5).map((row, i) => <tr key={row.strategy}><td><span className={`rank-number ${i === 0 ? "rank-first" : ""}`}>{String(i + 1).padStart(2, "0")}</span></td><td><div className="table-strategy">{row.strategy.split(" (")[0].split(" → ").map((compound, index) => <span key={`${compound}-${index}`}><CompoundTag compound={compound.toUpperCase()} />{index < row.strategy.split(" (")[0].split(" → ").length - 1 && <span className="strategy-arrow">→</span>}</span>)}</div></td><td className="time-cell">{fmt(row.totalTime)} <small>SEC</small></td><td className="muted-cell">{fmt(row.p10)}–{fmt(row.p90)}s</td><td className="blue-text">{fmt(row.winProbability * 100, 0)}%</td><td><span className={`status-pill ${i === 0 ? "status-best" : ""}`}>{i === 0 ? "TOP RANKED" : "SIMULATED"}</span></td></tr>)}</tbody></table>{strategies.length === 0 && <p className="empty-table">No strategy fits this race and tyre-set allocation.</p>}</div>
              </Card>
              {page === "Strategy Simulator" && <div className="dashboard-grid simulator-charts"><Card className="chart-card"><SectionHeading title="Race-time ranking" note="Predicted finish time · top 10 strategies" /><ResponsiveContainer width="100%" height={280}><BarChart data={strategies.slice(0, 10)} layout="vertical" margin={{ left: 22, right: 16 }}><CartesianGrid stroke="#272d38" horizontal={false} /><XAxis type="number" tick={{ fill: "#858d9d", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="strategy" width={175} tick={{ fill: "#b8beca", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip {...chartTooltip} /><Bar dataKey="totalTime" name="Race time (s)" radius={[0, 4, 4, 0]}>{strategies.slice(0, 10).map((_, i) => <Cell key={i} fill={i === 0 ? "#e10600" : "#4c5260"} />)}</Bar></BarChart></ResponsiveContainer></Card><Card className="chart-card"><SectionHeading title="Cumulative race pace" note="Predicted lap-time trend for the leading plan" /><ResponsiveContainer width="100%" height={280}><AreaChart data={race.laps.slice(0, raceLaps).filter((lap) => lap.driver === race.drivers[0])}><defs><linearGradient id="paceGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#e10600" stopOpacity={0.28} /><stop offset="95%" stopColor="#e10600" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="#272d38" vertical={false} /><XAxis dataKey="lapNumber" tick={{ fill: "#858d9d", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#858d9d", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip {...chartTooltip} /><Area type="monotone" dataKey="lapTime" stroke="#e10600" fill="url(#paceGradient)" /></AreaChart></ResponsiveContainer></Card></div>}
            </>}

            {page === "Race Explorer" && <><div className="filter-bar"><label>DRIVER<select value={selectedDriver} onChange={(e) => setSelectedDriver(e.target.value)}><option>All drivers</option>{race.drivers.map((driver) => <option key={driver}>{driver}</option>)}</select></label><div className="explorer-count"><Activity size={15} />{shownLaps.length} laps shown</div></div><Card className="chart-card explorer-chart"><SectionHeading title="Lap time by race lap" note="Each marker is one lap · compound colours remain consistent" /><ResponsiveContainer width="100%" height={390}><ScatterChart margin={{ top: 16, right: 20, bottom: 20, left: 0 }}><CartesianGrid stroke="#272d38" vertical={false} /><XAxis type="number" dataKey="lapNumber" name="Race lap" tick={{ fill: "#858d9d", fontSize: 11 }} label={{ value: "RACE LAP", position: "insideBottom", fill: "#858d9d", fontSize: 10, offset: -10 }} /><YAxis type="number" dataKey="lapTime" name="Lap time" tick={{ fill: "#858d9d", fontSize: 11 }} domain={["dataMin - 1", "dataMax + 1"]} /><Tooltip {...chartTooltip} cursor={{ strokeDasharray: "3 3" }} /><Legend /><Scatter data={shownLaps.filter((l) => l.compound === "SOFT")} name="Soft" fill={compoundColors.SOFT} /><Scatter data={shownLaps.filter((l) => l.compound === "MEDIUM")} name="Medium" fill={compoundColors.MEDIUM} /><Scatter data={shownLaps.filter((l) => l.compound === "HARD")} name="Hard" fill={compoundColors.HARD} /></ScatterChart></ResponsiveContainer></Card><Card className="ranking-card"><SectionHeading title="Lap data" note="Timing context is aligned from OpenF1 track-status, interval and position feeds. A dash means no value was published for that lap." /><div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>DRIVER</th><th>RACE LAP</th><th>LAP TIME</th><th>COMPOUND</th><th>TYRE AGE</th><th>TRACK STATUS</th><th>GAP AHEAD</th><th>POSITION</th><th>POSITION CHANGE</th></tr></thead><tbody>{shownLaps.slice(0, 30).map((lap, i) => <tr key={`${lap.driver}-${lap.lapNumber}-${i}`}><td>{lap.driver}</td><td>{lap.lapNumber}</td><td className="time-cell">{fmt(lap.lapTime, 3)} <small>SEC</small></td><td><CompoundTag compound={lap.compound} /></td><td>{lap.tyreLife}</td><td>{trackStatusLabel(lap.trackStatus)}</td><td>{lap.trafficGapSeconds == null ? "—" : `${fmt(lap.trafficGapSeconds, 3)} s`}</td><td>{lap.position ?? "—"}</td><td>{lap.positionChange == null ? "—" : `${lap.positionChange > 0 ? "+" : ""}${lap.positionChange}`}</td></tr>)}</tbody></table></div></Card></>}

            {page === "Tyre Degradation" && <><Card className="chart-card large-chart"><SectionHeading title="How tyre age changes lap pace" note="Mean lap time by compound and tyre age" /><ResponsiveContainer width="100%" height={410}><LineChart data={race.degradation}><CartesianGrid stroke="#272d38" vertical={false} /><XAxis dataKey="tyreAge" type="number" domain={[1, "dataMax"]} tick={{ fill: "#858d9d", fontSize: 11 }} label={{ value: "TYRE AGE (LAPS)", position: "insideBottom", fill: "#858d9d", fontSize: 10, offset: -4 }} /><YAxis dataKey="lapTime" domain={["dataMin - 1", "dataMax + 1"]} tick={{ fill: "#858d9d", fontSize: 11 }} /><Tooltip {...chartTooltip} /><Legend /><Line data={race.degradation.filter((v) => v.compound === "SOFT")} dataKey="lapTime" name="Soft" stroke={compoundColors.SOFT} dot={false} connectNulls /><Line data={race.degradation.filter((v) => v.compound === "MEDIUM")} dataKey="lapTime" name="Medium" stroke={compoundColors.MEDIUM} dot={false} connectNulls /><Line data={race.degradation.filter((v) => v.compound === "HARD")} dataKey="lapTime" name="Hard" stroke={compoundColors.HARD} dot={false} connectNulls /></LineChart></ResponsiveContainer><p className="chart-caption">Fuel load is not fully corrected in this MVP; use the curves to explore patterns, not infer definitive compound performance.</p></Card><div className="compound-cards">{(["SOFT", "MEDIUM", "HARD"] as const).map((compound) => { const points = race.degradation.filter((point) => point.compound === compound); const slope = points.length > 1 ? (points[points.length - 1].lapTime - points[0].lapTime) / (points[points.length - 1].tyreAge - points[0].tyreAge) : 0; return <Card key={compound} className="compound-card"><CompoundTag compound={compound} /><strong>{slope >= 0 ? "+" : ""}{fmt(slope, 3)}<small> s / lap</small></strong><p>Observed lap-time change per additional tyre-age lap.</p></Card>; })}</div></>}

            {page === "Model Performance" && <><div className="kpi-grid model-kpis"><Card className="kpi-card"><div className="kpi-top"><span>VALIDATION METHOD</span><ShieldCheck size={17} /></div><div className="model-kpi-value">Driver holdout</div><div className="kpi-foot">No random lap split</div></Card><Card className="kpi-card"><div className="kpi-top"><span>SELECTED MODEL</span><BarChart3 size={17} /></div><div className="model-kpi-value">{race.bestModel}</div><div className="kpi-foot">Lowest MAE available</div></Card><Card className="kpi-card"><div className="kpi-top"><span>TEST SET</span><Flag size={17} /></div><div className="model-kpi-value">{race.drivers.at(-1) ?? "—"}</div><div className="kpi-foot">Held-out driver</div></Card></div><Card className="ranking-card"><SectionHeading title="Model comparison" note="Lap-time error metrics on the held-out driver" /><div className="strategy-table-wrap"><table className="strategy-table"><thead><tr><th>MODEL</th><th>MAE (SEC)</th><th>RMSE (SEC)</th><th>R²</th><th>ROLE</th></tr></thead><tbody>{Object.entries(race.metrics).map(([name, metric]) => <tr key={name}><td className="model-name">{name}</td><td>{fmt(metric.MAE, 3)}</td><td>{fmt(metric.RMSE, 3)}</td><td>{fmt(metric.R2, 3)}</td><td><span className={`status-pill ${name === race.bestModel ? "status-best" : ""}`}>{name === race.bestModel ? "SELECTED" : "COMPARATOR"}</span></td></tr>)}</tbody></table></div></Card><div className="dashboard-grid"><Card className="chart-card"><SectionHeading title="Model MAE" note="Lower is better · seconds" /><ResponsiveContainer width="100%" height={270}><BarChart data={Object.entries(race.metrics).map(([name, m]) => ({ name, mae: m.MAE }))}><CartesianGrid stroke="#272d38" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#b8beca", fontSize: 11 }} /><YAxis tick={{ fill: "#858d9d", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Bar dataKey="mae" name="MAE (s)" radius={[5, 5, 0, 0]}>{Object.entries(race.metrics).map(([name], i) => <Cell key={name} fill={name === race.bestModel ? "#e10600" : ["#81899a", "#b3bac6", "#686f7d"][i % 3]} />)}</Bar></BarChart></ResponsiveContainer></Card><Card className="note-card"><div className="note-icon"><ShieldCheck size={20} /></div><h3>Validation without leakage</h3><p>Models are evaluated on a held-out driver rather than a random lap split. This reduces contamination from neighbouring laps in the training data.</p><div className="note-callout"><ArrowDownRight size={15} />Synthetic fallback data is useful for exercising the application, not for claiming real-world model performance.</div></Card></div></>}

            {page === "Methodology" && <><Card className="method-hero"><div className="method-icon"><Activity size={22} /></div><div><span className="title-kicker">FROM TIMING DATA TO RACE PACE</span><h2>A transparent strategy workflow.</h2><p>Explore the assumptions behind the model before trusting any recommendation.</p></div></Card><div className="method-grid">{[{ number: "01", title: "Collect & clean", body: "FastF1 supplies lap timing, tyre compound and age, weather, track status, and timing gaps. Pit laps, neutralised laps, missing times, and extreme outliers are filtered." }, { number: "02", title: "Fit race factors", body: "Team, traffic pressure, observed passes, compound, tyre age, fuel proxy, and weather are model features. Weather trend is applied lap by lap." }, { number: "03", title: "Validate by driver", body: "A driver is held out from training. MAE, RMSE, and R² summarize held-out lap-time predictions; they also calibrate uncertainty." }, { number: "04", title: "Simulate uncertainty", body: "Monte Carlo scenarios vary safety-car timing, traffic encounters, overtaking, weather noise, and model error. Tyre inventory constrains legal candidate stints." }].map((item) => <Card className="method-step" key={item.number}><span className="step-number">{item.number}</span><h3>{item.title}</h3><p>{item.body}</p></Card>)}</div><Card className="limitations-card"><div className="note-icon"><CircleHelp size={19} /></div><div><h3>Modelled factors still have uncertainty</h3><p>Traffic and overtaking responses are correlations in historical lap data, not a full track-position or passing simulation. Safety-car probabilities and tyre-set counts are scenario assumptions; actual allocations, incident timing, and red-flag restarts are not forecast. Team pace is estimated from the selected event and does not generalize reliably across seasons. Recommendations are educational simulations—not race engineering advice.</p></div></Card></>}

            <footer className="footer"><span><span className="footer-dot" />MODEL STATUS: READY</span><span>F1 STRATEGY LAB <b>·</b> BUILT FOR EXPLORATION</span></footer>
          </div>}
          {!race && !loading && !eventsLoading && !error && <div className="empty-state"><Activity size={28} /><h2>No race data available</h2><p>Check that the Python model service is running, then retry.</p></div>}
        </div>
      </main>
    </div>
  );
}
