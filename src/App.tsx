import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Info } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RTooltip,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  Cell,
} from "recharts";

/**
 * Vigenère Lab — IoC + Kasiski + Column Explorer (Shift Wheel)
 * Teaching-first, no auto-solve. Adds a per-column Caesar lab with chi-square scoring
 * and an interactive key composer + plaintext preview.
 */

// English letter frequencies (A..Z)
const ENG = [
  0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966,
  0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987,
  0.06327, 0.09056, 0.02758, 0.00978, 0.0236, 0.0015, 0.01974, 0.00074,
];

function normalizeAZ(s: string): string { return (s || "").toUpperCase().replace(/[^A-Z]/g, ""); }

function ioc(s: string): number {
  if (s.length < 2) return 0;
  const c = new Array(26).fill(0);
  for (let i = 0; i < s.length; i++) c[s.charCodeAt(i) - 65]++;
  const N = s.length; let num = 0; for (const x of c) num += x * (x - 1);
  return num / (N * (N - 1));
}

function splitColumns(s: string, k: number): string[] { const cols = Array.from({ length: k }, () => ""); for (let i = 0; i < s.length; i++) cols[i % k] += s[i]; return cols; }

function iocSweep(s: string, maxK: number): { k: number; ioc: number }[] {
  const out: { k: number; ioc: number }[] = [];
  for (let k = 1; k <= maxK; k++) { const cols = splitColumns(s, k); const mean = cols.reduce((acc, col) => acc + ioc(col), 0) / k; out.push({ k, ioc: Number(mean.toFixed(5)) }); }
  return out;
}

// Repeated n-grams with positions
function findRepeats(s: string, n: number) {
  const map: Record<string, number[]> = {};
  for (let i = 0; i <= s.length - n; i++) { const g = s.slice(i, i + n); (map[g] ||= []).push(i); }
  return Object.entries(map).filter(([, pos]) => pos.length >= 2).sort((a, b) => b[1].length - a[1].length).map(([gram, positions]) => ({ gram, positions }));
}

function distancesFromPositions(positions: number[], successiveOnly = false): number[] {
  const out: number[] = [];
  if (successiveOnly) { for (let i = 0; i < positions.length - 1; i++) out.push(positions[i + 1] - positions[i]); return out; }
  for (let i = 0; i < positions.length; i++) for (let j = i + 1; j < positions.length; j++) out.push(positions[j] - positions[i]);
  return out;
}

function factorVotes(distances: number[], limit: number): Record<number, number> {
  const votes: Record<number, number> = {};
  for (const d of distances) { for (let f = 2; f <= Math.min(limit, d); f++) { if (d % f === 0) votes[f] = (votes[f] || 0) + 1; } }
  return votes;
}

function highlightOccurrences(text: string, gram: string): JSX.Element[] {
  if (!gram) return [<span key="whole">{text}</span>];
  const out: JSX.Element[] = []; const n = gram.length;
  for (let i = 0; i < text.length;) { const chunk = text.slice(i, i + n); if (chunk === gram) { out.push(<mark key={i} className="rounded bg-yellow-200 px-0.5 py-0">{chunk}</mark>); i += n; } else { out.push(<span key={i}>{text[i]}</span>); i++; } }
  return out;
}

// Chi-square score for a column under shift k (lower is better). We "undo" shift to compare against ENG.
function chiForShift(col: string, k: number): number {
  const N = col.length; if (N === 0) return 1e9; const freq = new Array(26).fill(0);
  for (const ch of col) { const i = (ch.charCodeAt(0) - 65 - k + 26) % 26; freq[i]++; }
  let chi = 0; for (let i = 0; i < 26; i++) { const E = ENG[i] * N; const O = freq[i]; chi += (O - E) * (O - E) / (E || 1); }
  return chi;
}

function shiftToLetter(s: number | null): string { return s === null ? "?" : String.fromCharCode(65 + ((s % 26 + 26) % 26)); }
function letterToShift(ch: string): number { return (ch.toUpperCase().charCodeAt(0) - 65 + 26) % 26; }

// Partially decode original text: apply known shifts; unknowns become "·".
function decodePartial(original: string, keyShifts: (number | null)[], k: number): string {
  if (!original || k <= 0) return original;
  let j = 0; let out = "";
  for (const ch of original) {
    if (/[A-Za-z]/.test(ch)) {
      const upper = ch.toUpperCase();
      const s = keyShifts[j % k];
      if (s === null || s === undefined) { out += "·"; j++; continue; }
      const p = String.fromCharCode(((upper.charCodeAt(0) - 65 - s + 26) % 26) + 65);
      out += p; j++;
    } else { out += ch; }
  }
  return out;
}

export default function VigenereIoCKasiskiMiniApp() {
  // INPUT
  const [raw, setRaw] = useState<string>("");
  const [ngram, setNgram] = useState(3);
  const [pairwise, setPairwise] = useState(true);
  const [maxK, setMaxK] = useState(30);
  const [limitFactors, setLimitFactors] = useState(30);
  const [selectedGram, setSelectedGram] = useState<string>("");
  const [selectedK, setSelectedK] = useState<number | null>(null);

  // COLUMNS / KEY
  const [colIndex, setColIndex] = useState(0);
  const [keyShifts, setKeyShifts] = useState<(number | null)[]>([]);
  const [manualKeyInput, setManualKeyInput] = useState("");

  const sanitized = useMemo(() => normalizeAZ(raw), [raw]);

  const repeats = useMemo(() => findRepeats(sanitized, ngram), [sanitized, ngram]);
  const kasiskiVotes = useMemo(() => {
    const counts: Record<number, number> = {};
    for (const r of repeats) {
      const dists = distancesFromPositions(r.positions, !pairwise ? true : false);
      const v = factorVotes(dists, limitFactors);
      for (const [k, c] of Object.entries(v)) counts[Number(k)] = (counts[Number(k)] || 0) + (c as number);
    }
    return counts;
  }, [repeats, pairwise, limitFactors]);

  const kasiskiData = useMemo(() => Object.entries(kasiskiVotes)
    .map(([k, v]) => ({ k: Number(k), votes: v as number }))
    .filter((d) => d.k <= maxK)
    .sort((a, b) => a.k - b.k), [kasiskiVotes, maxK]);

  const iocData = useMemo(() => iocSweep(sanitized, maxK), [sanitized, maxK]);

  const workingK = useMemo(() => {
    if (selectedK) return selectedK;
    // fallback: top kasiski by votes or IoC by value
    const k1 = [...kasiskiData].sort((a, b) => b.votes - a.votes)[0]?.k;
    const k2 = [...iocData].sort((a, b) => b.ioc - a.ioc)[0]?.k;
    return k1 || k2 || 1;
  }, [selectedK, kasiskiData, iocData]);

  // Maintain keyShifts length with workingK
  useEffect(() => {
    setKeyShifts((prev) => Array.from({ length: workingK }, (_, i) => (prev[i] ?? null)));
    setColIndex(0);
  }, [workingK]);

  const cols = useMemo(() => splitColumns(sanitized, Math.max(1, workingK)), [sanitized, workingK]);
  const activeCol = cols[colIndex] || "";

  const chiData = useMemo(() => {
    const arr = Array.from({ length: 26 }, (_, s) => ({ shift: s, chi: Number(chiForShift(activeCol, s).toFixed(3)) }));
    return arr;
  }, [activeCol]);

  const bestShift = useMemo(() => chiData.reduce((best, d) => (d.chi < best.chi ? d : best), chiData[0] || { shift: 0, chi: 1e9 }).shift, [chiData]);

  function setShiftFor(index: number, value: number) {
    setKeyShifts((prev) => {
      const next = [...prev]; next[index] = ((value % 26) + 26) % 26; return next;
    });
  }

  function keyString(): string { return keyShifts.map(shiftToLetter).join(""); }

  // Manual key entry (e.g., JACK) → fill shifts
  function applyManualKey() {
    if (!manualKeyInput) return;
    const letters = manualKeyInput.replace(/[^A-Za-z]/g, "").toUpperCase();
    if (!letters) return;
    const shifts = Array.from({ length: workingK }, (_, i) => letterToShift(letters[i % letters.length]));
    setKeyShifts(shifts);
  }

  const preview = useMemo(() => decodePartial(raw, keyShifts, Math.max(1, workingK)), [raw, keyShifts, workingK]);

  const examplePaste = `CVP{BCT34W!}KQJKFZQNZEXXQKQZZ...`;

  return (
    <div className="p-6 space-y-4 text-[15px]">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-bold tracking-tight">Vigenère Lab — IoC, Kasiski & Columns</h1>
        <div className="text-xs text-muted-foreground flex items-center gap-1"><Info size={14}/> No auto‑solve. Explore the evidence.</div>
      </div>

      <Tabs defaultValue="input" className="w-full">
        <TabsList className="grid grid-cols-4 w-full md:w-[720px]">
          <TabsTrigger value="input">Input</TabsTrigger>
          <TabsTrigger value="kasiski">Kasiski</TabsTrigger>
          <TabsTrigger value="ioc">IoC</TabsTrigger>
          <TabsTrigger value="columns">Columns</TabsTrigger>
        </TabsList>

        {/* INPUT */}
        <TabsContent value="input" className="space-y-4">
          <Card className="border-foreground/10">
            <CardHeader><CardTitle className="text-base">Ciphertext</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <Textarea value={raw} onChange={(e) => setRaw(e.target.value)} placeholder="Paste your ciphertext here (any characters)." className="min-h-[140px]" />
              <div className="flex flex-wrap items-center gap-3">
                <Button variant="secondary" onClick={() => setRaw(examplePaste)}>Paste sample</Button>
                <Badge variant="outline" className="rounded-full">Original length: {raw.length}</Badge>
                <Badge variant="outline" className="rounded-full">Analyzed (A–Z) length: {sanitized.length}</Badge>
              </div>
              <Separator />
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-sm">Sanitized (A–Z)</Label>
                  <div className="mt-2 p-3 rounded-xl bg-muted/40 border text-sm leading-6 break-words">{highlightOccurrences(sanitized, selectedGram)}</div>
                </div>
                <div>
                  <Label className="text-sm">Original (for reference)</Label>
                  <div className="mt-2 p-3 rounded-xl bg-muted/40 border text-sm leading-6 break-words whitespace-pre-wrap">{raw || <span className="text-muted-foreground">(empty)</span>}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* KASISKI */}
        <TabsContent value="kasiski" className="space-y-4">
          <div className="grid lg:grid-cols-2 gap-4">
            <Card className="border-foreground/10">
              <CardHeader><CardTitle className="text-base">Repeated n‑grams</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-3"><Label>n‑gram</Label><Slider className="w-40" value={[ngram]} min={3} max={6} step={1} onValueChange={(v) => setNgram(v[0])} /><Badge className="rounded-full">{ngram}</Badge></div>
                  <div className="flex items-center gap-2"><Switch id="pair" checked={pairwise} onCheckedChange={setPairwise} /><Label htmlFor="pair">Use all pairs</Label></div>
                </div>
                <div className="max-h-64 overflow-auto rounded-lg border">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 sticky top-0"><tr><th className="px-2 py-2 text-left">n‑gram</th><th className="px-2 py-2 text-left">positions</th><th className="px-2 py-2 text-left">distances</th></tr></thead>
                    <tbody>
                      {repeats.length === 0 && (<tr><td colSpan={3} className="px-3 py-8 text-center text-muted-foreground">No repeats found yet. Try a longer text or lower n.</td></tr>)}
                      {repeats.map((r) => { const d = distancesFromPositions(r.positions, !pairwise ? true : false); return (
                        <tr key={r.gram} className={`hover:bg-muted/40 cursor-pointer ${selectedGram === r.gram ? "bg-accent/10" : ""}`} onClick={() => setSelectedGram(r.gram)}>
                          <td className="px-2 py-1 font-mono text-xs">{r.gram}</td>
                          <td className="px-2 py-1 text-xs">{r.positions.join(", ")}</td>
                          <td className="px-2 py-1 text-xs">{d.join(", ")}</td>
                        </tr> ); })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card className="border-foreground/10">
              <CardHeader><CardTitle className="text-base">Key‑length candidates (Kasiski votes)</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-3">
                  <Label>Consider factors up to</Label><Slider className="w-40" value={[limitFactors]} min={10} max={40} step={1} onValueChange={(v) => setLimitFactors(v[0])} /><Badge className="rounded-full">{limitFactors}</Badge>
                  <Label className="ml-4">Max K for charts</Label><Slider className="w-40" value={[maxK]} min={10} max={40} step={1} onValueChange={(v) => setMaxK(v[0])} /><Badge className="rounded-full">{maxK}</Badge>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={kasiskiData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="k" tickFormatter={(v) => `${v}`} />
                      <YAxis allowDecimals={false} />
                      <RTooltip formatter={(v: any) => [v, "votes"]} labelFormatter={(label) => `k=${label}`} />
                      <Bar dataKey="votes" onClick={(data) => setSelectedK((data as any).k)}>
                        {kasiskiData.map((entry, idx) => (<Cell key={`cell-${idx}`} opacity={selectedK === entry.k ? 0.9 : 1} />))}
                      </Bar>
                      {selectedK && <ReferenceLine x={selectedK} stroke="#e74c3c" strokeDasharray="3 3" />}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* IoC */}
        <TabsContent value="ioc" className="space-y-4">
          <Card className="border-foreground/10">
            <CardHeader><CardTitle className="text-base">IoC sweep (average per‑column IoC vs. key length)</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <div className="grid md:grid-cols-3 gap-4">
                <div className="md:col-span-2 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={iocData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="k" />
                      <YAxis domain={[0.03, 0.08]} />
                      <RTooltip formatter={(v: any) => [v, "IoC"]} labelFormatter={(label) => `k=${label}`} />
                      <Line type="monotone" dataKey="ioc" dot={false} />
                      <ReferenceLine y={0.066} stroke="#0A3A40" strokeDasharray="4 2" />
                      <ReferenceLine y={0.038} stroke="#999" strokeDasharray="4 2" />
                      {selectedK && <ReferenceLine x={selectedK} stroke="#e74c3c" strokeDasharray="3 3" />}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2">
                  <div className="text-sm"><div className="font-medium">References</div><div className="text-muted-foreground">English ≈ 0.066</div><div className="text-muted-foreground">Random ≈ 0.038</div></div>
                  <Separator />
                  <div className="text-sm font-medium">Pick a working k</div>
                  <div className="flex items-center gap-2">{selectedK ? (<Badge className="rounded-full" variant="default">k = {selectedK}</Badge>) : (<Badge className="rounded-full" variant="secondary">none selected</Badge>)}<Button variant="secondary" size="sm" onClick={() => setSelectedK(null)}>Clear</Button></div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* COLUMNS */}
        <TabsContent value="columns" className="space-y-4">
          <Card className="border-foreground/10">
            <CardHeader><CardTitle className="text-base">Column Explorer (k = {workingK})</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              {/* NEW: manual key-length (k) control */}
              <div className="flex flex-wrap items-center gap-3">
                <Label className="text-sm">Key length (k)</Label>
                <Slider className="w-64" value={[workingK]} min={1} max={40} step={1} onValueChange={(v) => setSelectedK(v[0])} />
                <Input type="number" className="w-20" value={workingK} min={1} max={40}
                  onChange={(e) => {
                    const val = parseInt(e.target.value || '1', 10);
                    if (!Number.isNaN(val)) setSelectedK(Math.max(1, Math.min(40, val)));
                  }} />
                <Button size="sm" variant="ghost" onClick={() => setSelectedK(null)}>Auto (use Kasiski/IoC)</Button>
              </div>

              <Separator />

              <div className="flex flex-wrap items-center gap-3">
                <Label className="text-sm">Manual key</Label>
                <Input value={manualKeyInput} onChange={(e) => setManualKeyInput(e.target.value)} placeholder="e.g., JACK" className="w-40" />
                <Button size="sm" variant="secondary" onClick={applyManualKey}>Apply</Button>
                <Button size="sm" variant="ghost" onClick={() => setKeyShifts(Array.from({ length: workingK }, () => null))}>Reset</Button>
                <Badge variant="outline" className="rounded-full">Key: {keyString()}</Badge>
              </div>

              {/* Key composer */}
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: workingK }, (_, i) => (
                  <button key={i} onClick={() => setColIndex(i)} className={`px-3 py-2 rounded-2xl text-sm border shadow-sm ${colIndex === i ? "bg-accent/10 border-accent" : "bg-muted/40"}`}>
                    <div className="font-mono text-base">{shiftToLetter(keyShifts[i] ?? null)}</div>
                    <div className="text-[11px] text-muted-foreground">pos {i + 1}</div>
                  </button>
                ))}
              </div>

              {/* Column details */}
              <div className="grid md:grid-cols-5 gap-4">
                <div className="md:col-span-2">
                  <div className="text-sm font-medium mb-2">Selected column #{colIndex + 1}</div>
                  <div className="p-3 rounded-xl bg-muted/40 border font-mono text-sm leading-6 break-words">
                    {activeCol || <span className="text-muted-foreground">(empty)</span>}
                  </div>
                </div>
                <div className="md:col-span-3 space-y-3">
                  <div className="flex items-center gap-3">
                    <Label className="text-sm">Shift</Label>
                    <Slider className="w-64" value={[keyShifts[colIndex] ?? 0]} min={0} max={25} step={1} onValueChange={(v) => setShiftFor(colIndex, v[0])} />
                    <Badge className="rounded-full">{shiftToLetter(keyShifts[colIndex] ?? 0)} ({keyShifts[colIndex] ?? 0})</Badge>
                    <Button size="sm" variant="secondary" onClick={() => setShiftFor(colIndex, bestShift)}>Set best (χ²)</Button>
                  </div>

                  {/* A–Z quick picker */}
                  <div className="flex flex-wrap gap-1">
                    {Array.from({ length: 26 }, (_, s) => (
                      <Badge key={s} variant={keyShifts[colIndex] === s ? "default" : "secondary"} className="cursor-pointer" onClick={() => setShiftFor(colIndex, s)}>
                        {String.fromCharCode(65 + s)}
                      </Badge>
                    ))}
                  </div>

                  {/* Chi-square chart */}
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chiData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="shift" tickFormatter={(v) => String.fromCharCode(65 + v)} />
                        <YAxis />
                        <RTooltip formatter={(v: any) => [v, "χ²"]} labelFormatter={(label) => `shift ${String.fromCharCode(65 + (label as number))}`} />
                        <Bar dataKey="chi" onClick={(d) => setShiftFor(colIndex, (d as any).shift)}>
                          {chiData.map((entry, idx) => (<Cell key={`c-${idx}`} opacity={(keyShifts[colIndex] ?? -1) === entry.shift ? 0.9 : 1} />))}
                        </Bar>
                        <ReferenceLine x={bestShift} stroke="#e74c3c" strokeDasharray="3 3" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Preview */}
              <div>
                <Label className="text-sm">Plaintext preview (unknown letters shown as ·)</Label>
                <div className="mt-2 p-3 rounded-xl bg-muted/40 border font-mono text-sm leading-6 whitespace-pre-wrap break-words">
                  {preview}
                </div>
              </div>
            </CardContent>
          </Card>
          <div className="text-xs text-muted-foreground">Tip: Lower χ² ≈ closer to English. Use the best χ² as a starting point, then refine by eye and with cribs.</div>
        </TabsContent>
      </Tabs>

      {/* Footer tips */}
      <div className="text-xs text-muted-foreground leading-relaxed">Kasiski distances share multiples of the key length. IoC rises near the true k because each column behaves like a monoalphabetic substitution. Column Explorer lets you pick per-position shifts that minimize χ² and make real words appear.</div>
    </div>
  );
}

