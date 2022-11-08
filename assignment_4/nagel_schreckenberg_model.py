import numpy as np
from gif_tool import GifTool


class NagelSchreckenbergModel:
    EMPTY_SPOT = -9

    def __init__(self, density: float, slowing_prob: float, road_len: int = 100, max_velocity: int = 5,
                 gif_tool: None | GifTool = None):
        self.density = density
        self.slowing_prob = slowing_prob
        self.road_len = road_len
        self.cars_num = int(density * road_len)
        self.max_velocity = max_velocity

        self.line = None
        self.create_line()

        self.cars = None
        self.distance = []
        # what about initial velocity?

        self.gif_tool = gif_tool

    def _update_cars(self):
        self.cars = np.where(self.line != self.EMPTY_SPOT)[0]
        return self.cars

    def _update_distance(self):
        self.distance = []
        for i, car in enumerate(self.cars[:-1]):
            self.distance.append(self.cars[i+1]-car)

        # last car (rounding boundary conditions)
        self.distance.append(self.cars[0] - self.cars[-1] + self.road_len)

    def create_line(self):
        self.line = np.ones(self.road_len - self.cars_num) * self.EMPTY_SPOT
        self.line = np.append(self.line, np.ones(self.cars_num))
        np.random.shuffle(self.line)

    def single_step(self):
        self._update_cars()
        self._update_distance()
        # print('cars:', self.cars)

        self._acceleration()
        self._slowing_down()
        self._randomization()
        # print('random line:', self.line)
        self._car_motion()
        # print('motion     :',self.line,'\n')

    def play(self, iter_num: int):
        if self.gif_tool:
            self.gif_tool.save_pic(data=self.line, title=f'initial state')
        for i in range(iter_num):
            self.single_step()
            if self.gif_tool:
                self.gif_tool.save_pic(data=self.line, title=f'iteration {i}')

    def _acceleration(self):
        for car in self.cars:
            if self.line[car] < self.max_velocity:
                self.line[car] += 1

    def _slowing_down(self):
        for i, car in enumerate(self.cars):
            max_vel = self.distance[i] - 1
            if max_vel < self.line[car]:
                self.line[car] = max_vel

    def _randomization(self):
        for car in self.cars:
            if self.line[car] == 0:
                # cannot slow down car which is not moving
                continue

            if np.random.random() < self.slowing_prob:
                self.line[car] -= 1

    def _car_motion(self):
        for car in self.cars:
            # get car velocity
            vel = self.line[car]

            # move car
            if vel > 0:
                future_pos = int((car + vel) % self.road_len)
                self.line[future_pos] = vel
                # clear previous position of the car
                self.line[car] = self.EMPTY_SPOT

    def get_avg_velocity(self):
        return np.mean(self.line[self.line != self.EMPTY_SPOT])


if __name__ == "__main__":
    params = {
        "density": 0.1,
        "slowing_prob": 0.1,
        "road_len": 100,
        "max_velocity": 5
    }
    gif_tool = GifTool()

    a = NagelSchreckenbergModel(**params, gif_tool=GifTool())

    # print(a._update_cars())
    a.play(20)
    gif_tool.save_gif('testing.gif',duration=1)