import sys
import pygame as pg
import os
import time


def play_sound(sound_file):
    clock = pg.time.Clock()
    try:
        pg.mixer.init()
        pg.mixer.music.load(sound_file)
        print("Music file {} loaded!".format(sound_file))
    except pg.error:
        print("File {} not found! {}".format(sound_file, pg.get_error()))
        return

    pg.mixer.music.play()

    while pg.mixer.music.get_busy():
        clock.tick(30)


if __name__ == '__main__':
    play_sound(os.getcwd() + os.sep + 'car-honk.mp3')
