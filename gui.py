import pygame
import chess
from io import BytesIO
import cairosvg

ANALYSIS_HEIGHT = 120   # px added below the existing button bar

class ChessGUI:
    def __init__(self, game_engine, board_size=400, flipped=False):
        pygame.init()
        self.game = game_engine
        self.board_size = board_size
        self.square_size = board_size // 8
        self.flipped = flipped  # True = Black's perspective (rank 1 at top, h-file left)
        self.screen = pygame.display.set_mode((board_size, board_size + 50 + ANALYSIS_HEIGHT))
        pygame.display.set_caption("Chess Vision")
        self.piece_images = self.load_piece_images()
        self.selected_square = None
        self.valid_moves = []
        self._analysis_result = None   # (lines, depth) from EngineWorker.get_result()

    def load_piece_images(self):
        pieces = {}
        piece_symbols = {
            'P': chess.PAWN,
            'N': chess.KNIGHT,
            'B': chess.BISHOP,
            'R': chess.ROOK,
            'Q': chess.QUEEN,
            'K': chess.KING
        }

        for color in ['w', 'b']:
            for symbol, piece_type in piece_symbols.items():
                chess_piece = chess.Piece(piece_type, chess.WHITE if color == 'w' else chess.BLACK)
                svg = chess.svg.piece(chess_piece)

                png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
                img = pygame.image.load(BytesIO(png))

                key = f"{color}{symbol}"
                pieces[key] = pygame.transform.smoothscale(
                    img, (self.square_size, self.square_size))
        return pieces

    def _screen_coords(self, chess_file, chess_rank):
        """Convert chess file/rank (0-indexed) to screen col/row."""
        if self.flipped:
            return 7 - chess_file, chess_rank
        else:
            return chess_file, 7 - chess_rank

    def _chess_square_from_screen(self, screen_col, screen_row):
        """Convert screen col/row to chess square index."""
        if self.flipped:
            return chess.square(7 - screen_col, screen_row)
        else:
            return chess.square(screen_col, 7 - screen_row)

    def update_analysis(self, result):
        """Called each frame with the latest result from EngineWorker.get_result()."""
        self._analysis_result = result

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                pygame.draw.rect(
                    self.screen, color,
                    (col * self.square_size, row * self.square_size,
                     self.square_size, self.square_size)
                )

                if self._chess_square_from_screen(col, row) in self.valid_moves:
                    s = pygame.Surface((self.square_size, self.square_size))
                    s.set_alpha(100)
                    s.fill((124, 252, 0))
                    self.screen.blit(s, (col * self.square_size, row * self.square_size))

        for square in chess.SQUARES:
            piece = self.game.board.piece_at(square)
            if piece:
                piece_code = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
                col, row = self._screen_coords(chess.square_file(square), chess.square_rank(square))
                self.screen.blit(
                    self.piece_images[piece_code],
                    (col * self.square_size, row * self.square_size)
                )

        font = pygame.font.SysFont('Arial', 16)
        moves_text = font.render(self.game.get_move_history(), True, (0, 0, 0))
        self.screen.blit(moves_text, (10, self.board_size + 10))

        pygame.draw.rect(self.screen, (200, 200, 200),
                        (self.board_size - 150, self.board_size + 10, 70, 30))
        reset_text = font.render("Reset", True, (0, 0, 0))
        self.screen.blit(reset_text, (self.board_size - 140, self.board_size + 15))

        pygame.draw.rect(self.screen, (200, 200, 200),
                        (self.board_size - 70, self.board_size + 10, 60, 30))
        copy_text = font.render("Copy", True, (0, 0, 0))
        self.screen.blit(copy_text, (self.board_size - 60, self.board_size + 15))

    def draw_analysis_panel(self, engine_available):
        """Draw the Stockfish analysis panel below the button bar."""
        panel_y = self.board_size + 50
        panel_w = self.board_size
        panel_h = ANALYSIS_HEIGHT

        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), (0, panel_y, panel_w, panel_h))

        font_header = pygame.font.SysFont('Arial', 13)
        font_line   = pygame.font.SysFont('Courier New', 14)

        if not engine_available:
            msg = font_header.render("Stockfish not found — install it and add to PATH", True, (160, 160, 160))
            self.screen.blit(msg, (8, panel_y + 8))
            return

        # Header row
        if self._analysis_result is None:
            header = font_header.render("Stockfish  analysing...", True, (160, 160, 160))
        else:
            _, depth = self._analysis_result
            header = font_header.render(f"Stockfish  depth {depth}", True, (160, 160, 160))
        self.screen.blit(header, (8, panel_y + 6))

        if self._analysis_result is None:
            return

        lines, _ = self._analysis_result
        line_y = panel_y + 26

        for i, (score_str, moves_str, _depth) in enumerate(lines[:3]):
            # Score colour: green positive, red negative, grey for mate
            if score_str.startswith("M") or score_str.startswith("-M"):
                score_color = (255, 215, 0)      # gold for mate
            elif score_str.startswith("+"):
                score_color = (100, 220, 100)    # green
            elif score_str.startswith("-"):
                score_color = (220, 100, 100)    # red
            else:
                score_color = (200, 200, 200)

            rank_surf  = font_line.render(f"{i+1}.", True, (140, 140, 140))
            score_surf = font_line.render(f"{score_str:<7}", True, score_color)
            moves_surf = font_line.render(moves_str, True, (220, 220, 220))

            x = 8
            self.screen.blit(rank_surf,  (x,      line_y))
            x += rank_surf.get_width() + 4
            self.screen.blit(score_surf, (x,      line_y))
            x += score_surf.get_width() + 4
            self.screen.blit(moves_surf, (x,      line_y))

            line_y += 28

    def handle_click(self, pos):
        x, y = pos

        if y > self.board_size:
            if self.board_size - 150 <= x <= self.board_size - 80:
                return "RESET"
            elif self.board_size - 70 <= x <= self.board_size - 10:
                return "COPY"
            return None

        col, row = x // self.square_size, y // self.square_size
        square = self._chess_square_from_screen(col, row)

        if self.selected_square is None:
            piece = self.game.board.piece_at(square)
            if piece and piece.color == self.game.board.turn:
                self.selected_square = square
                self.valid_moves = [
                    move.to_square for move in self.game.board.legal_moves
                    if move.from_square == square
                ]

        else:
            move_str = f"{chess.square_name(self.selected_square)}{chess.square_name(square)}"

            piece = self.game.board.piece_at(self.selected_square)
            if piece.piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                move_str += "q"

            if self.game.make_move(move_str):
                self.selected_square = None
                self.valid_moves = []
                return "MOVE"
            elif square in self.valid_moves:
                self.selected_square = square
            else:
                self.selected_square = None
                self.valid_moves = []

        return None
