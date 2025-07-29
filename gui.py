import pygame
import chess
from io import BytesIO
import cairosvg

class ChessGUI:
    def __init__(self, game_engine, board_size=400):
        pygame.init()
        self.game = game_engine
        self.board_size = board_size
        self.square_size = board_size // 8
        self.screen = pygame.display.set_mode((board_size, board_size + 50))
        pygame.display.set_caption("Chess Vision")
        
        self.piece_images = self.load_piece_images()
        
        self.selected_square = None
        self.valid_moves = []
    
    def load_piece_images(self):
        pieces = {}
        for color in ['w', 'b']:
            for piece in ['P', 'N', 'B', 'R', 'Q', 'K']:
                svg = chess.svg.piece(chess.Piece.from_symbol(piece), color=color)
                
                png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
                img = pygame.image.load(BytesIO(png))
                
                pieces[f"{color}{piece}"] = pygame.transform.smoothscale(
                    img, (self.square_size, self.square_size))
        return pieces
    
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                pygame.draw.rect(
                    self.screen, color,
                    (col * self.square_size, row * self.square_size,
                     self.square_size, self.square_size)
                )
                
                if chess.square(col, 7-row) in self.valid_moves:
                    s = pygame.Surface((self.square_size, self.square_size))
                    s.set_alpha(100)
                    s.fill((124, 252, 0))
                    self.screen.blit(s, (col * self.square_size, row * self.square_size))
        
        for square in chess.SQUARES:
            piece = self.game.board.piece_at(square)
            if piece:
                piece_code = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
                col, row = chess.square_file(square), 7 - chess.square_rank(square)
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
    
    def handle_click(self, pos):
        x, y = pos
        
        if y > self.board_size:
            if self.board_size - 150 <= x <= self.board_size - 80:
                return "RESET"
            elif self.board_size - 70 <= x <= self.board_size - 10:
                return "COPY"
            return None
        
        col, row = x // self.square_size, 7 - (y // self.square_size)
        square = chess.square(col, row)
        
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
            elif square in self.valid_moves:
                self.selected_square = square
            else:
                self.selected_square = None
                self.valid_moves = []
        
        return None