import threading
import chess
import chess.engine
import os


def _format_score(score_obj, board):
    """Return a display string for a PovScore, always from White's perspective."""
    pov = score_obj.white()
    if pov.is_mate():
        m = pov.mate()
        return f"M{m}" if m > 0 else f"-M{abs(m)}"
    cp = pov.score()
    return f"{'+' if cp >= 0 else ''}{cp / 100:.2f}"


def _build_lines(multipv, board):
    """Convert analysis.multipv to a list of (score_str, moves_str, depth) tuples."""
    lines = []
    for pv_info in multipv:
        if "score" not in pv_info or "pv" not in pv_info:
            continue
        score_str = _format_score(pv_info["score"], board)
        depth = pv_info.get("depth", 0)

        san_moves = []
        tmp = board.copy()
        for move in pv_info["pv"][:5]:
            try:
                san_moves.append(tmp.san(move))
                tmp.push(move)
            except Exception:
                break

        lines.append((score_str, " ".join(san_moves), depth))
    return lines


class EngineWorker:
    """Runs Stockfish in a background daemon thread.

    Thread-safe interface:
      request_analysis(fen)  — call from main thread to start/replace analysis
      get_result()           — call from main thread each frame; returns latest lines
      is_available()         — False if the Stockfish binary was not found
      quit()                 — clean shutdown; call before pygame.quit()
    """

    def __init__(self, path=None):
        self._path = path or os.environ.get("STOCKFISH_PATH", "stockfish")
        self._lock = threading.Lock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._pending_fen = None   # written by main thread, read by worker
        self._result = None        # (lines, depth) written by worker, read by main thread
        self._available = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── public API (main thread) ────────────────────────────────────────────

    def request_analysis(self, fen):
        """Start (or replace) analysis of the given FEN position."""
        self._pending_fen = fen     # GIL makes this atomic
        self._request_event.set()

    def get_result(self):
        """Return (lines, depth) or None. lines = list of (score, moves, depth)."""
        with self._lock:
            return self._result

    def is_available(self):
        return self._available

    def quit(self):
        self._stop_event.set()
        self._request_event.set()   # unblock the worker if it's waiting
        self._thread.join(timeout=2)

    # ── worker thread ───────────────────────────────────────────────────────

    def _run(self):
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self._path)
            self._available = True
        except (FileNotFoundError, OSError):
            return  # leaves _available = False

        try:
            while not self._stop_event.is_set():
                self._request_event.wait()
                self._request_event.clear()

                if self._stop_event.is_set():
                    break

                fen = self._pending_fen
                if fen is None:
                    continue

                try:
                    board = chess.Board(fen)
                except Exception:
                    continue

                try:
                    with engine.analysis(
                        board,
                        chess.engine.Limit(depth=20),
                        multipv=3
                    ) as analysis:
                        for _ in analysis:
                            if self._stop_event.is_set() or self._pending_fen != fen:
                                break

                            lines = _build_lines(analysis.multipv, board)
                            if lines:
                                depth = lines[0][2]
                                with self._lock:
                                    self._result = (lines, depth)

                except chess.engine.EngineTerminatedError:
                    break
                except Exception:
                    pass

        finally:
            try:
                engine.quit()
            except Exception:
                pass
