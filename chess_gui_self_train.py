import math, time, chess, pygame, pandas as pd
from chess_game import model_vs_human

pygame.init()
X, Y = 800, 800
scrn = pygame.display.set_mode((X, Y))
clock = pygame.time.Clock()

BLUE = (50,255,255)
light_square = (238, 238, 210)
dark_square  = (118, 150, 86)
font = pygame.font.SysFont(None, 40)

b = chess.Board()

path = r"images/"

pieces = {'p': pygame.image.load(path+'b_pawn.png').convert_alpha(), 
        'n': pygame.image.load(path+'b_knight.png').convert_alpha(),
        'b': pygame.image.load(path+'b_bishop.png').convert_alpha(), 
        'r': pygame.image.load(path+'b_rook.png').convert_alpha(), 
        'q': pygame.image.load(path+'b_queen.png').convert_alpha(), 
        'k': pygame.image.load(path+'b_king.png').convert_alpha(), 
        'P': pygame.image.load(path+'w_pawn.png').convert_alpha(), 
        'N': pygame.image.load(path+'w_knight.png').convert_alpha(), 
        'B': pygame.image.load(path+'w_bishop.png').convert_alpha(), 
        'R': pygame.image.load(path+'w_rook.png').convert_alpha(), 
        'Q': pygame.image.load(path+'w_queen.png').convert_alpha(), 
        'K': pygame.image.load(path+'w_king.png').convert_alpha(), }

def draw_board(scrn, board):
    for row in range(8):
        for col in range(8):
            color = light_square if (row + col) % 2 == 0 else dark_square
            pygame.draw.rect(scrn, color, pygame.Rect(col * 100, row * 100, 100, 100))
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            x = (i % 8) * 100
            y = 700 - (i // 8) * 100
            scrn.blit(pieces[str(piece)], (x, y))
    pygame.display.flip()

def draw_rating_buttons():
    rate_pos_btn = pygame.Rect(100, 750, 200, 40)
    rate_neg_btn = pygame.Rect(500, 750, 200, 40)
    pygame.draw.rect(scrn, (0,200,0), rate_pos_btn)
    pygame.draw.rect(scrn, (200,0,0), rate_neg_btn)
    scrn.blit(font.render("good (+1)", True, (255,255,255)), (130, 755))
    scrn.blit(font.render("bad (-1)", True, (255,255,255)), (540, 755))
    pygame.display.flip()
    return rate_pos_btn, rate_neg_btn

def main(board, human_color=chess.WHITE):
    data = {"fen": [], "move": [], "score": []}
    running = True
    index_moves, moves = [], []

    while running:
        clock.tick(30)
        draw_board(scrn, board)

        # bot turn
        if board.turn != human_color and not board.outcome():
            fen = board.fen()
            ai_move = model_vs_human(board, 2)
            if ai_move:
                board.push(ai_move)
                draw_board(scrn, board)

                # rating buttons
                rate_pos_btn, rate_neg_btn = draw_rating_buttons()
                waiting = True
                score = 0
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            pos = pygame.mouse.get_pos()
                            if rate_pos_btn.collidepoint(pos):
                                score = 1
                                waiting = False
                            elif rate_neg_btn.collidepoint(pos):
                                score = -1
                                waiting = False
                    clock.tick(30)

                data["fen"].append(fen)
                data["move"].append(str(ai_move))
                data["score"].append(score)
            continue

        # human turn
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = (math.floor(pos[0] / 100), math.floor(pos[1] / 100))
                index = (7 - square[1]) * 8 + square[0]

                if index in index_moves:
                    move = moves[index_moves.index(index)]
                    board.push(move)
                    index_moves.clear()
                else:
                    piece = board.piece_at(index)
                    if not piece or piece.color != human_color:
                        continue
                    all_moves = list(board.legal_moves)
                    moves = [m for m in all_moves if m.from_square == index]
                    index_moves = [m.to_square for m in moves]

                    draw_board(scrn, board)
                    for m in moves:
                        tx, ty = (m.to_square % 8) * 100, (7 - m.to_square // 8) * 100
                        pygame.draw.rect(scrn, BLUE, pygame.Rect(tx, ty, 100, 100), 5)
                    pygame.display.flip()

        if board.outcome():
            print(board.outcome())
            running = False

    pygame.quit()
    old_df = pd.read_csv("chess_positions_bot.csv")
    df = pd.DataFrame(data)
    df = pd.concat([old_df, df])
    df.to_csv("chess_positions_bot.csv", index=False)


if __name__ == "__main__":
    main(b)