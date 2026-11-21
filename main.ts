// Sokoban solver: push-optimal A* over the level's rooms.
//
// A generated level is a hallway with independent rooms off it: a box pushed
// out of its room can never come back, so each room is solved on its own and
// the level's optimum is the sum of its rooms' optima. Inside a room the
// search is A* over pushes: a state is the boxes plus the player's reachable
// region (named by its lowest square), the cost is pushes made, and the
// estimate is the cheapest assignment of boxes to goals by shortest-path
// distance through open floor. Dead states are pruned: boxes on squares from
// which no push can ever reach a goal (pulled back from the goals), boxes
// frozen in a 2x2 block of walls and boxes, and boxes frozen in place by walls
// and other frozen boxes. Everything is budgeted by work, never by the clock,
// so two runs write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work budgets, in states: never the clock, so runs agree byte for byte. */
const ROOM_STATE_BUDGET = 250_000;
const LEVEL_STATE_BUDGET = 500_000;

const DIRS: Array<[number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  if (player < 0) throw new Error('no player in level');
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Squares from which a box can be pushed onto some goal, pulled back from the goals over open floor. */
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue = b.goals.slice();
  for (const g of b.goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * b.w + dx;
      const y = x - d;
      if (!b.wall[y] && !b.wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood the player's reachable region from `from` with `occupied` in the way;
 * marks it in `seen` with `stamp` and returns its lowest square.
 */
function reach(
  b: Board,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const w = b.w;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const q0 = p - w;
    const q1 = p + w;
    const q2 = p - 1;
    const q3 = p + 1;
    if (!b.wall[q0] && !occupied[q0] && seen[q0] !== stamp) { seen[q0] = stamp; stack[n++] = q0; }
    if (!b.wall[q1] && !occupied[q1] && seen[q1] !== stamp) { seen[q1] = stamp; stack[n++] = q1; }
    if (!b.wall[q2] && !occupied[q2] && seen[q2] !== stamp) { seen[q2] = stamp; stack[n++] = q2; }
    if (!b.wall[q3] && !occupied[q3] && seen[q3] !== stamp) { seen[q3] = stamp; stack[n++] = q3; }
  }
  return low;
}

/** Whether the player can walk from `from` onto some floor square of `row`, with `boxes` in the way. */
function playerReachesRow(
  b: Board,
  boxes: number[],
  from: number,
  row: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): boolean {
  const w = b.w;
  const occupied = new Uint8Array(w * b.h);
  for (const cell of boxes) occupied[cell] = 1;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (Math.floor(p / w) === row) return true;
    const q0 = p - w;
    const q1 = p + w;
    const q2 = p - 1;
    const q3 = p + 1;
    if (!b.wall[q0] && !occupied[q0] && seen[q0] !== stamp) { seen[q0] = stamp; stack[n++] = q0; }
    if (!b.wall[q1] && !occupied[q1] && seen[q1] !== stamp) { seen[q1] = stamp; stack[n++] = q1; }
    if (!b.wall[q2] && !occupied[q2] && seen[q2] !== stamp) { seen[q2] = stamp; stack[n++] = q2; }
    if (!b.wall[q3] && !occupied[q3] && seen[q3] !== stamp) { seen[q3] = stamp; stack[n++] = q3; }
  }
  return false;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(b: Board, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const w = b.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const c = a + 1;
    const d = a + w;
    const e = d + 1;
    if (!(b.wall[a] || occupied[a])) continue;
    if (!(b.wall[c] || occupied[c])) continue;
    if (!(b.wall[d] || occupied[d])) continue;
    if (!(b.wall[e] || occupied[e])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d]) || (occupied[e] && !goal[e])) return true;
  }
  return false;
}

/** Whether any box in `boxes` is frozen in place by walls and other frozen boxes while off its goal. */
function frozenOffGoal(b: Board, boxes: number[], goal: Uint8Array): boolean {
  const w = b.w;
  const frozen = new Uint8Array(w * b.h);
  let changed = true;
  while (changed) {
    changed = false;
    for (const cell of boxes) {
      if (frozen[cell]) continue;
      const blocked = (c: number): boolean => b.wall[c] === 1 || frozen[c] === 1;
      // A box can move left only with its left square free and its right square
      // standable, and so on: it is stuck when every move is barred on one side
      // or the other, along both axes at once.
      if ((blocked(cell - 1) || blocked(cell + 1)) && (blocked(cell - w) || blocked(cell + w))) {
        frozen[cell] = 1;
        changed = true;
      }
    }
  }
  for (const cell of boxes) if (frozen[cell] && !goal[cell]) return true;
  return false;
}

/** All orderings of n goal indices, for the assignment estimate. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Uint8Array(n);
  const rec = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = 1;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = 0;
    }
  };
  rec();
  return out;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(b: Board, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.w * b.h).fill(-1);
  const how: string[] = new Array(b.w * b.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const c = p + dy * b.w + dx;
      if (b.wall[c] || occupied[c] || prev[c] !== -1) continue;
      prev[c] = p;
      how[c] = letter;
      queue.push(c);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = (how[p] ?? '') + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Found {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal A* for one board. A state is the sorted boxes plus the player's
 * reachable region (named by its lowest square). `exitRow`, when given, is a
 * row the player must be able to reach once every box is on a goal, so the
 * player can leave the room just solved. Returns null past `maxStates`.
 */
function solvePush(b: Board, maxStates: number, exitRow: number | null): Found | null {
  const size = b.w * b.h;
  const goal = new Uint8Array(size);
  for (const g of b.goals) goal[g] = 1;
  const live = liveSquares(b);
  for (const box of b.boxes) if (!live[box]) return null;
  const n = b.boxes.length;
  const solved = (boxes: number[]): boolean => boxes.every((c) => goal[c] === 1);
  if (solved(b.boxes)) return { pushes: 0, moves: '' };

  // Shortest-path distance from each goal to every square, over open floor.
  const dist: Int32Array[] = b.goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi];
      for (const [dx, dy] of DIRS) {
        const c = p + dy * b.w + dx;
        if (!b.wall[c] && d[c] < 0) {
          d[c] = d[p] + 1;
          q.push(c);
        }
      }
    }
    return d;
  });
  const perms = permutations(n);
  const estimate = (boxes: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = dist[perm[i]][boxes[i]];
        if (d < 0) {
          s = Infinity;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const seen2 = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let stamp2 = 0;

  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const box of boxes) k = k * size + box;
    return k;
  };

  const bx: number[][] = [b.boxes.slice()];
  const pl: number[] = [b.player];
  const lo: number[] = [];
  const par: number[] = [-1];
  const pFrom: number[] = [-1];
  const pTo: number[] = [-1];
  for (const box of b.boxes) occupied[box] = 1;
  lo.push(reach(b, occupied, b.player, seen, ++stamp, stack));
  for (const box of b.boxes) occupied[box] = 0;
  const best = new Map<number, number>();
  best.set(keyOf(b.boxes, lo[0]), 0);
  const closed = new Set<number>();

  // Binary heap on [f, -g, index]: lowest f first, then deepest, then earliest.
  const heap: Array<[number, number, number]> = [];
  const heapLess = (a: [number, number, number], c: [number, number, number]): boolean =>
    a[0] < c[0] || (a[0] === c[0] && (a[1] < c[1] || (a[1] === c[1] && a[2] < c[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0 && last !== undefined) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1], heap[c])) c++;
        if (!heapLess(heap[c], heap[i])) break;
        const tmp = heap[i];
        heap[i] = heap[c];
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  heapPush([estimate(b.boxes), 0, 0]);
  let expanded = 0;
  while (heap.length > 0) {
    const [, negG, s] = heapPop();
    const g = -negG;
    const sKey = keyOf(bx[s], lo[s]);
    if (closed.has(sKey)) continue;
    if (best.get(sKey) !== g) continue;
    closed.add(sKey);
    const boxes = bx[s];
    if (solved(boxes)) {
      if (exitRow === null || playerReachesRow(b, boxes, pl[s], exitRow, seen, ++stamp, stack)) {
        return { pushes: g, moves: rebuild(b, bx, pl, par, pFrom, pTo, s) };
      }
      continue; // solved, but the player is stuck: keep looking for one that lets it out
    }
    if (expanded >= maxStates) return null;
    expanded++;
    for (const box of boxes) occupied[box] = 1;
    reach(b, occupied, pl[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const cell = boxes[bi];
      for (const [dx, dy, , pushLetter] of DIRS) {
        const d = dy * b.w + dx;
        const t = cell + d;
        if (seen[cell - d] !== here) continue;
        if (b.wall[t] || occupied[t] || !live[t]) continue;
        occupied[cell] = 0;
        occupied[t] = 1;
        const dead = frozenBlock(b, occupied, goal, t);
        let next: number[] | null = null;
        let low = -1;
        if (!dead) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, c) => a - c);
          if (frozenOffGoal(b, next, goal)) {
            next = null;
          } else {
            low = reach(b, occupied, cell, seen2, ++stamp2, stack);
          }
        }
        occupied[t] = 0;
        occupied[cell] = 1;
        if (next === null) continue;
        const key = keyOf(next, low);
        const ng = g + 1;
        const known = best.get(key);
        if (known !== undefined && known <= ng) continue;
        best.set(key, ng);
        const idx = bx.length;
        bx.push(next);
        pl.push(cell);
        lo.push(low);
        par.push(s);
        pFrom.push(cell);
        pTo.push(t);
        heapPush([ng + estimate(next), -ng, idx]);
      }
    }
    for (const box of boxes) occupied[box] = 0;
  }
  return null;
}

/** The moves of the push chain ending at `goalState`, walking to each push first. */
function rebuild(
  b: Board,
  bx: number[][],
  pl: number[],
  par: number[],
  pFrom: number[],
  pTo: number[],
  goalState: number,
): string {
  const chain: number[] = [];
  for (let s = goalState; s > 0; s = par[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = pl[0];
  let boxes = bx[0];
  const occupied = new Uint8Array(b.w * b.h);
  for (const s of chain) {
    const from = pFrom[s];
    const to = pTo[s];
    const d = to - from;
    for (const box of boxes) occupied[box] = 1;
    moves += walkTo(b, occupied, player, from - d);
    for (const box of boxes) occupied[box] = 0;
    const dir = DIRS.find(([dx, dy]) => dy * b.w + dx === d);
    moves += dir ? dir[3] : '?';
    player = from;
    boxes = bx[s];
  }
  return moves;
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a
 * box there reaches no goal) out of the floor, and each part left that holds a
 * box is a room. Null when some part holds more or fewer boxes than goals, or
 * a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(b: Board): Room[] | null {
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.w * b.h);
  for (let i = row * b.w; i < (row + 1) * b.w; i++) {
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((cell) => cut[cell])) return null;
  const part = new Int32Array(b.w * b.h).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < b.w * b.h; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const c = squares[i] + dy * b.w + dx;
        if (!b.wall[c] && !cut[c] && part[c] < 0) {
          part[c] = rooms.length;
          squares.push(c);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const cell of b.boxes) rooms[part[cell]].boxes.push(cell);
  for (const g of b.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The level solved room by room: each room's push-optimal solution in turn, the other rooms' boxes standing as walls. */
function solveLevel(b: Board): string {
  const rooms = roomsOf(b);
  if (rooms === null) {
    const whole = solvePush(b, LEVEL_STATE_BUDGET, null);
    return whole === null ? '' : whole.moves;
  }
  const hallwayRow = Math.floor(b.player / b.w);
  const done: Room[] = [];
  const pending: Room[] = rooms.slice();
  let player = b.player;
  let moves = '';
  const solveRoom = (room: Room, exitRow: number | null): Found | null => {
    const wall = b.wall.slice();
    for (const other of pending) {
      if (other === room) continue;
      for (const cell of other.boxes) wall[cell] = 1;
    }
    for (const other of done) {
      for (const cell of other.goals) wall[cell] = 1;
    }
    const sub: Board = { w: b.w, h: b.h, wall, goals: room.goals.slice(), boxes: room.boxes.slice(), player };
    return solvePush(sub, ROOM_STATE_BUDGET, exitRow);
  };
  while (pending.length > 0) {
    let progressed = false;
    for (let i = 0; i < pending.length; i++) {
      const room = pending[i];
      const lastOne = pending.length === 1;
      const sol = solveRoom(room, lastOne ? null : hallwayRow);
      if (sol !== null) {
        moves += sol.moves;
        player = endOf(b.w, player, sol.moves);
        done.push(room);
        pending.splice(i, 1);
        progressed = true;
        break;
      }
      if (!lastOne) {
        // No exit-compatible solution: this room can only go last, if at all.
        const free = solveRoom(room, null);
        if (free !== null) continue;
      }
      return '';
    }
    if (!progressed) {
      const room = pending[0];
      const sol = solveRoom(room, null);
      if (sol === null) return '';
      moves += sol.moves;
      player = endOf(b.w, player, sol.moves);
      done.push(room);
      pending.shift();
    }
  }
  return moves;
}

for (const id of readdirSync(IN_DIR).sort()) {
  let moves = '';
  try {
    const level = parseLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
    moves = solveLevel(level);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
