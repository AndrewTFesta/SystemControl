import pygame
import pygame.locals


def load_tile_table(filename, width, height):
    image = pygame.image.load(filename).convert()
    image_width, image_height = image.get_size()
    tile_table = []
    for tile_x in range(0, int(image_width / width)):
        line = []
        tile_table.append(line)
        for tile_y in range(0, int(image_height / height)):
            rect = (tile_x * width, tile_y * height, width, height)
            line.append(image.subsurface(rect))
    return tile_table

class MazeGame:
    DISPLAY_WIDTH = 400
    DISPLAY_HEIGHT = 400

    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((self.DISPLAY_HEIGHT, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('Maze Game')
        self.running = False
        return

    def run(self):
        self.running = True
        while self.running:
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
        pygame.quit()
        return

def main():
    maze_game = MazeGame()
    maze_game.run()
    return

    # pygame.init()
    # screen = pygame.display.set_mode((128, 98))
    # screen.fill((255, 255, 255))
    # table = load_tile_table(os.path.join(IMAGES_DIR, 'ground.png'), 24, 16)
    # for x, row in enumerate(table):
    #     for y, tile in enumerate(row):
    #         screen.blit(tile, (x * 32, y * 24))
    # pygame.display.flip()
    # while pygame.event.wait().type != pygame.locals.QUIT:
    #     pass
    # return


if __name__ == '__main__':
    main()
