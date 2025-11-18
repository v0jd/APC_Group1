#!/usr/bin/env python3
"""
Mouse Archery — Manual Power + Fullscreen (Pygame)

Controls
--------
- Move mouse to aim (archer follows cursor at bottom).
- Mouse wheel: adjust power (0–100%) before shooting.
- Left click: shoot immediately using current power.
- F: toggle fullscreen/windowed (state is preserved).
- P: pause, R: restart, ESC: quit.

Notes
-----
- No hold-to-charge; power persists until you adjust it.
- Uses only shapes (no assets).
"""

import math
import random
import sys
import pygame

# ----------------------------
# Game configuration constants
# ----------------------------
WIDTH, HEIGHT = 960, 600        # windowed size
FPS = 60

ARROW_SPEED_MIN = 8.0           # speed at 0% power
ARROW_SPEED_MAX = 22.0          # speed at 100% power
SCROLL_STEP = 0.10              # 10% per scroll notch
GRAVITY = 0.22                  # gravity for arc

ARROW_MAX = 15                  # starting arrows
ARROW_REWARD_ON_HIT = 1         # gain arrows on hit

TARGET_MIN_SPEED = 1.2
TARGET_MAX_SPEED = 2.6
TARGET_SPAWN_EVERY = 75         # frames
TARGET_RADIUS = 22

PLAYER_BOTTOM_OFFSET = 60       # distance from bottom
PLAYER_WIDTH = 60
PLAYER_HEIGHT = 28

BACKGROUND = (20, 24, 36)
UI_COLOR = (230, 230, 240)
BOW_COLOR = (245, 200, 120)
ARROW_COLOR = (220, 220, 220)
TARGET_COLORS = [(255, 99, 132), (54, 162, 235), (255, 206, 86),
                 (75, 192, 192), (153, 102, 255), (255, 159, 64)]

# ----------------------------
# Helper math
# ----------------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def vec_from_points(x1, y1, x2, y2, desired_mag):
    dx = x2 - x1
    dy = y2 - y1
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0, -desired_mag
    scale = desired_mag / mag
    return dx * scale, dy * scale

def rotate_point(px, py, cx, cy, ang_rad):
    s, c = math.sin(ang_rad), math.cos(ang_rad)
    px, py = px - cx, py - cy
    rx = px * c - py * s
    ry = px * s + py * c
    return rx + cx, ry + cy

# ----------------------------
# Entity classes
# ----------------------------
class Arrow:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alive = True

    def update(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy
        if self.x < -50 or self.x > WIDTH + 50 or self.y < -50 or self.y > HEIGHT + 50:
            self.alive = False

    def angle(self):
        return math.atan2(self.vy, self.vx)

    def tip(self):
        return (self.x, self.y)

    def draw(self, surf):
        ang = self.angle()
        length = 28
        back_x = self.x - math.cos(ang) * length
        back_y = self.y - math.sin(ang) * length
        pygame.draw.line(surf, ARROW_COLOR, (back_x, back_y), (self.x, self.y), 3)
        tip = (self.x, self.y)
        left = rotate_point(self.x - 10, self.y - 5, self.x, self.y, ang)
        right = rotate_point(self.x - 10, self.y + 5, self.x, self.y, ang)
        pygame.draw.polygon(surf, ARROW_COLOR, [tip, left, right])

class Target:
    def __init__(self):
        side = random.choice(["left", "right"])
        y = random.randint(100, HEIGHT - 220)
        if side == "left":
            x = -TARGET_RADIUS - 10
            vx = random.uniform(TARGET_MIN_SPEED, TARGET_MAX_SPEED)
        else:
            x = WIDTH + TARGET_RADIUS + 10
            vx = -random.uniform(TARGET_MIN_SPEED, TARGET_MAX_SPEED)
        self.x = x
        self.y = y
        self.vx = vx
        self.phase = random.random() * math.tau
        self.color = random.choice(TARGET_COLORS)
        self.alive = True

    def update(self, t):
        self.x += self.vx
        self.y += math.sin(self.phase + t * 0.05) * 0.6
        if self.x < -40 or self.x > WIDTH + 40:
            self.alive = False

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), TARGET_RADIUS)
        pygame.draw.circle(surf, (255, 255, 255), (int(self.x), int(self.y)), TARGET_RADIUS//2, 2)
        pygame.draw.circle(surf, (255, 255, 255), (int(self.x), int(self.y)), 4)

    def hit_test(self, x, y):
        return (self.x - x) ** 2 + (self.y - y) ** 2 <= (TARGET_RADIUS + 4) ** 2

class Player:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - PLAYER_BOTTOM_OFFSET

    def update_from_mouse(self):
        mx, my = pygame.mouse.get_pos()
        self.x = clamp(mx, PLAYER_WIDTH//2 + 10, WIDTH - PLAYER_WIDTH//2 - 10)
        self.y = HEIGHT - PLAYER_BOTTOM_OFFSET

    def draw(self, surf):
        rect = pygame.Rect(0, 0, PLAYER_WIDTH, PLAYER_HEIGHT)
        rect.center = (self.x, self.y)
        pygame.draw.rect(surf, BOW_COLOR, rect, border_radius=10)

        mx, my = pygame.mouse.get_pos()
        dx, dy = mx - self.x, my - self.y
        ang = math.atan2(dy, dx)
        r = 26
        p1 = (self.x + r * math.cos(ang + 1.2), self.y + r * math.sin(ang + 1.2))
        p2 = (self.x + r * math.cos(ang - 1.2), self.y + r * math.sin(ang - 1.2))
        pygame.draw.line(surf, (160, 120, 70), p1, p2, 3)
        pygame.draw.line(surf, (240, 220, 200), p1, p2, 1)
        guide_len = 40
        gx = self.x + guide_len * math.cos(ang)
        gy = self.y + guide_len * math.sin(ang)
        pygame.draw.line(surf, (120, 140, 180), (self.x, self.y), (gx, gy), 2)

# ----------------------------
# Main game class
# ----------------------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Mouse Archery — Manual Power + Fullscreen")
        self.flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), self.flags)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.bigfont = pygame.font.SysFont("arial", 48, bold=True)
        self.fullscreen = False
        self.reset()

    def reset(self):
        self.player = Player()
        self.arrows = []
        self.targets = []
        self.score = 0
        self.arrows_left = ARROW_MAX
        self.frame = 0
        self.paused = False
        self.game_over = False
        self.best_score = getattr(self, "best_score", 0)
        self.power = 0.5   # Manual power (0..1), default 50%

    def spawn_target(self):
        self.targets.append(Target())

    def shoot(self):
        if self.game_over or self.paused:
            return
        if self.arrows_left <= 0:
            return
        self.arrows_left -= 1
        px, py = self.player.x, self.player.y
        mx, my = pygame.mouse.get_pos()
        speed = ARROW_SPEED_MIN + (ARROW_SPEED_MAX - ARROW_SPEED_MIN) * self.power
        vx, vy = vec_from_points(px, py, mx, my, speed)
        self.arrows.append(Arrow(px, py, vx, vy))

    def adjust_power(self, delta_steps):
        self.power = clamp(self.power + delta_steps * SCROLL_STEP, 0.0, 1.0)

    def toggle_fullscreen(self):
        global WIDTH, HEIGHT
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pygame.display.Info()
            WIDTH, HEIGHT = info.current_w, info.current_h
            self.flags = pygame.FULLSCREEN
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), self.flags)
        else:
            WIDTH, HEIGHT = 960, 600
            self.flags = pygame.RESIZABLE
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), self.flags)
        self.player.y = HEIGHT - PLAYER_BOTTOM_OFFSET

    def update(self):
        if self.paused or self.game_over:
            return

        self.frame += 1
        self.player.update_from_mouse()

        if self.frame % TARGET_SPAWN_EVERY == 0:
            for _ in range(random.choice([1, 1, 2])):
                self.spawn_target()

        for a in self.arrows:
            a.update()
        self.arrows = [a for a in self.arrows if a.alive]

        for t in self.targets:
            t.update(self.frame)
        self.targets = [t for t in self.targets if t.alive]

        # Collisions
        for a in self.arrows:
            tipx, tipy = a.tip()
            for t in self.targets:
                if t.alive and t.hit_test(tipx, tipy):
                    t.alive = False
                    a.alive = False
                    self.score += 10
                    self.arrows_left += ARROW_REWARD_ON_HIT
                    break

        if self.arrows_left <= 0 and len(self.arrows) == 0:
            self.game_over = True
            self.best_score = max(self.best_score, self.score)

    def draw_grid(self, surf):
        gap = 40
        color = (30, 34, 48)
        for x in range(0, WIDTH, gap):
            pygame.draw.line(surf, color, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, gap):
            pygame.draw.line(surf, color, (0, y), (WIDTH, y))

    def draw_power_bar(self):
        bar_w, bar_h = 320, 16
        x = (WIDTH - bar_w) // 2
        y = HEIGHT - 50
        pygame.draw.rect(self.screen, (50, 60, 90), (x, y, bar_w, bar_h), border_radius=8)
        fill_w = int(bar_w * self.power)
        pygame.draw.rect(self.screen, (120, 190, 120), (x, y, fill_w, bar_h), border_radius=8)
        lbl = self.font.render("Power", True, (210, 220, 230))
        self.screen.blit(lbl, (x - 70, y - 2))

    def draw(self):
        self.screen.fill(BACKGROUND)
        self.draw_grid(self.screen)

        for t in self.targets:
            t.draw(self.screen)
        for a in self.arrows:
            a.draw(self.screen)
        self.player.draw(self.screen)

        ui = f"Score: {self.score}   Arrows: {self.arrows_left}   Best: {self.best_score}"
        txt = self.font.render(ui, True, UI_COLOR)
        self.screen.blit(txt, (16, 14))

        hint = self.font.render("Scroll: Power • Left click: Shoot • F: Fullscreen • P: Pause • R: Restart • ESC: Quit",
                                True, (170, 180, 195))
        self.screen.blit(hint, (16, HEIGHT - 30))

        self.draw_power_bar()

        if self.paused:
            shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 120))
            self.screen.blit(shade, (0, 0))
            ptxt = self.bigfont.render("PAUSED", True, (240, 240, 255))
            self.screen.blit(ptxt, ptxt.get_rect(center=(WIDTH//2, HEIGHT//2)))

        if self.game_over:
            shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 140))
            self.screen.blit(shade, (0, 0))
            over = self.bigfont.render("OUT OF ARROWS!", True, (255, 235, 235))
            self.screen.blit(over, over.get_rect(center=(WIDTH//2, HEIGHT//2 - 40)))
            scr = self.font.render(f"Final score: {self.score} • Best: {self.best_score}", True, (240, 240, 255))
            self.screen.blit(scr, scr.get_rect(center=(WIDTH//2, HEIGHT//2 + 10)))
            msg = self.font.render("Press R to play again", True, (220, 220, 235))
            self.screen.blit(msg, msg.get_rect(center=(WIDTH//2, HEIGHT//2 + 40)))

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_f:
                    self.toggle_fullscreen()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.shoot()
            elif event.type == pygame.MOUSEWHEEL:
                self.adjust_power(+1 if event.y > 0 else -1)

    def run(self):
        while True:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

def main():
    Game().run()

if __name__ == "__main__":
    main()
