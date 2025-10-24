# train_all.py
import os
from train import train

ATARI_49_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
    'BankHeist', 'BattleZone', 'BeamRider', 'Bowling', 'Boxing', 'Breakout',
    'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
    'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
    'Hero', 'IceHockey', 'JamesBond', 'Kangaroo', 'Krull', 'KungFuMaster',
    'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall',
    'Pong', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
    'Seaquest', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot',
    'Tutankham', 'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'Zaxxon'
]

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    for game in ATARI_49_GAMES:
        env_name = f"{game}NoFrameskip-v4"
        try:
            gymnasium.make(env_name)
            train(env_name, f"models/dqn_{game.lower()}.pth")
        except Exception as e:
            print(f"Skipping {game}: {e}")