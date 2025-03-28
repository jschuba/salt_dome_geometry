import math
import matplotlib.pyplot as plt
import pandas as pd


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def translate(self, x, y):
        self.x = self.x + x
        self.y = self.y + y
        return self

    def copy(self):
        return Point(self.x, self.y)

    def distance(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Square:
    def __init__(self, center: Point, edge_length: float):
        """
        :param center: The point at the center of the square
        :param edge_length: The full length of one edge.
        """
        self.center = center
        self.edge_length = edge_length

    @property
    def area(self):
        return self.edge_length * self.edge_length

    @property
    def perimeter(self):
        return 4 * self.edge_length

    def copy(self):
        return Square(self.center.copy(), self.edge_length)

    def translate(self, x, y):
        self.center.translate(x, y)
        return self

    @property
    def _half_edge_length(self):
        return self.edge_length / 2

    def exterior_points(self) -> list[Point]:
        return [self.lower_left, self.lower_right, self.upper_right, self.upper_left]

    @property
    def lower_left(self) -> Point:
        return self.center.copy().translate(-self._half_edge_length, -self._half_edge_length)

    @property
    def lower_right(self) -> Point:
        return self.center.copy().translate(self._half_edge_length, -self._half_edge_length)

    @property
    def upper_left(self) -> Point:
        return self.center.copy().translate(-self._half_edge_length, self._half_edge_length)

    @property
    def upper_right(self) -> Point:
        return self.center.copy().translate(self._half_edge_length, self._half_edge_length)

    def plot_to_ax(self, ax: plt.Axes, **kwargs) -> plt.Axes:
        pts = self.exterior_points()
        pts = pts + [pts[0]]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        ax.plot(xs, ys, **kwargs)
        return ax

    def inscribed_circle(self) -> "Circle":
        return Circle(self.center, self.edge_length / 2)

    def circumscribed_circle(self) -> "Circle":
        # The radius of the circumscribed circle is half the diagonal of the square
        diagonal = math.sqrt(2) * self.edge_length
        radius = diagonal / 2
        return Circle(self.center, radius)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Square):
            return False
        return self.center == other.center and self.edge_length == other.edge_length

    def __hash__(self):
        return hash((self.center, self.edge_length))

    def __repr__(self):
        return f"Square({self.center}, {self.edge_length})"


class Hexagon:
    def __init__(self, center: Point, radius: float):
        """
        Hexagon

        Note: The radius is the distance from the center to one of the outer points.  AKA the Circumradius.
        Use the following equation to get translate between the Circumradius and the Inradius:
        inradius = circumradius * sqrt(3) / 2


        :param center: A point at the center of the hexagon.
        :param radius: The radius is the distance from the center to one of the outer points.  AKA the Circumradius
        """
        self.center = center
        self.radius = radius

    @property
    def area(self):
        return (3 * math.sqrt(3) * self.radius ** 2) / 2

    @property
    def perimeter(self):
        return 6 * self.radius

    def copy(self):
        return Hexagon(self.center.copy(), self.radius)

    def translate(self, x, y):
        self.center.translate(x, y)
        return self

    def exterior_points(self) -> list[Point]:
        return [self.upper, self.upper_right, self.lower_right, self.lower, self.lower_left, self.upper_left]

    def inscribed_circle(self) -> "Circle":
        return Circle(self.center, self.radius * math.sqrt(3) / 2)

    def circumscribed_circle(self) -> "Circle":
        return Circle(self.center, self.radius)

    @property
    def upper(self) -> Point:
        return self.center.copy().translate(0, self.radius)

    @property
    def upper_right(self) -> Point:
        return self.center.copy().translate(self.radius * math.sqrt(3) / 2, self.radius / 2)

    @property
    def lower_right(self) -> Point:
        return self.center.copy().translate(self.radius * math.sqrt(3) / 2, -self.radius / 2)

    @property
    def lower(self) -> Point:
        return self.center.copy().translate(0, -self.radius)

    @property
    def lower_left(self) -> Point:
        return self.center.copy().translate(-self.radius * math.sqrt(3) / 2, -self.radius / 2)

    @property
    def upper_left(self) -> Point:
        return self.center.copy().translate(-self.radius * math.sqrt(3) / 2, self.radius / 2)

    def plot_to_ax(self, ax: plt.Axes, **kwargs) -> plt.Axes:
        pts = self.exterior_points()
        pts = pts + [pts[0]]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        ax.plot(xs, ys, **kwargs)
        return ax

    def __eq__(self, other) -> bool:
        if not isinstance(other, Hexagon):
            return False
        return self.center == other.center and self.radius == other.radius

    def __hash__(self):
        return hash((self.center, self.radius))

    def __repr__(self):
        return f"Hexagon({self.center}, {self.radius})"


class Circle:

    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    @property
    def area(self):
        return self.radius * self.radius * math.pi

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

    def exterior_points(self, n=25) -> list[Point]:
        l = list()
        for i in range(n):
            angle = i * 2 * math.pi / n
            x_offset = math.cos(angle) * self.radius
            y_offset = math.sin(angle) * self.radius
            l.append(self.center.copy().translate(x_offset, y_offset))
        return l

    def contains_points(self, p: list[Point] | Point) -> bool:
        if isinstance(p, Point):
            return self.center.distance(p) <= self.radius
        elif isinstance(p, list):
            return all(self.contains_points(p) for p in p)
        else:
            raise TypeError("Not a Point or list of Points")

    def plot_to_ax(self, ax: plt.Axes, n_points: int = 25, **kwargs) -> plt.Axes:
        if n_points <= 0:
            raise ValueError("n_points must be greater than zero")
        pts = self.exterior_points(n=n_points)
        pts = pts + [pts[0]]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        ax.plot(xs, ys, **kwargs)
        return ax

    def __eq__(self, other):
        if not isinstance(other, Circle):
            return False
        return self.center == other.center and self.radius == other.radius

    def __hash__(self):
        return hash((self.center, self.radius))

    def __repr__(self):
        return f"Circle({self.center}, {self.radius})"


def create_grid_of_squares(edge_length: float, lower_left_bound: Point, upper_right_bound: Point) -> list[list[Square]]:
    if not (lower_left_bound.x < upper_right_bound.x and lower_left_bound.y < upper_right_bound.y):
        raise ValueError("lower_left_bound must be lower and more left than upper_right_bound")

    if (upper_right_bound.x - lower_left_bound.x) % edge_length == 0:
        # The x bound is evenly divisible:
        start_center_x = lower_left_bound.x + edge_length / 2
    else:
        center_x = (lower_left_bound.x + upper_right_bound.x) / 2
        start_center_x = center_x - edge_length * math.floor(
            (center_x - lower_left_bound.x) / edge_length) + edge_length / 2
        print(center_x, start_center_x)
    if (upper_right_bound.y - lower_left_bound.y) % edge_length == 0:
        # The y bound is evenly divisible:
        start_center_y = lower_left_bound.y + edge_length / 2
    else:
        center_y = (lower_left_bound.y + upper_right_bound.y) / 2
        start_center_y = center_y - edge_length * math.floor(
            (center_y - lower_left_bound.y) / edge_length) + edge_length / 2

    sq = Square(Point(start_center_x, start_center_y), edge_length)
    row = [sq]
    while sq.center.x + 1.5 * edge_length <= upper_right_bound.x:
        sq = sq.copy().translate(edge_length, 0)
        row.append(sq)
    rows = [row]

    new_row = [p.copy() for p in row]
    while new_row[0].center.y + 1.5 * edge_length <= upper_right_bound.y:
        new_row = [p.copy().translate(0, edge_length) for p in new_row]
        rows.append(new_row)

    return rows


def create_grid_of_hexagons(radius: float, lower_left_bound: Point, upper_right_bound: Point) -> list[list[Hexagon]]:
    if not (lower_left_bound.x < upper_right_bound.x and lower_left_bound.y < upper_right_bound.y):
        raise ValueError("lower_left_bound must be lower and more left than upper_right_bound")

    # Calculate the horizontal and vertical spacing between centers of adjacent hexagons
    horizontal_spacing = radius * math.sqrt(3)  # Horizontal distance between centers of adjacent hexagons
    vertical_spacing = 1.5 * radius  # Vertical distance between centers of adjacent hexagons

    # Calculate the starting center point (adjusted to fit the grid inside the bounds)
    start_center_x = lower_left_bound.x + horizontal_spacing / 2
    start_center_y = lower_left_bound.y + vertical_spacing / 2

    # Generate the grid of hexagons
    row = []
    current_x = start_center_x + horizontal_spacing / 2

    # First row of hexagons
    while current_x + horizontal_spacing / 2 <= upper_right_bound.x:
        hexagon = Hexagon(Point(current_x, start_center_y), radius)
        row.append(hexagon)
        current_x += horizontal_spacing

    rows = [row]

    # Generate subsequent rows of hexagons
    current_y = start_center_y + vertical_spacing
    while current_y + vertical_spacing / 2 <= upper_right_bound.y:
        # Offset even rows by half the horizontal spacing
        if len(rows) % 2 == 0:  # Even row: offset horizontally
            current_x = start_center_x + horizontal_spacing / 2
        else:  # Odd row: start directly below the first column of the previous row
            current_x = start_center_x

        new_row = []
        while current_x + horizontal_spacing / 2 <= upper_right_bound.x:
            hexagon = Hexagon(Point(current_x, current_y), radius)
            new_row.append(hexagon)
            current_x += horizontal_spacing

        rows.append(new_row)
        current_y += vertical_spacing

    return rows


def plot_list_to_axis(l, ax, **kwargs):
    if hasattr(l, "plot_to_ax"):
        l.plot_to_ax(ax, **kwargs)
    elif isinstance(l, list):
        for itm in l:
            plot_list_to_axis(itm, ax, **kwargs)


if __name__ == "__main__":

    """
    ++++++++++++++++
    EXAMPLE USAGE:
    +++++++++++++++
    """
    salt_cavern_diameter = 200
    inter_cavern_spacing = salt_cavern_diameter * 4  # distance from cavern edge to edge

    inscribed_circle_radius = inter_cavern_spacing + salt_cavern_diameter / 2
    square_edge_length = inscribed_circle_radius * 2
    hexagon_circumradius = inscribed_circle_radius * 2 / math.sqrt(3)

    dome_circle_radius = 4000

    circ = Circle(Point(0, 0), dome_circle_radius)
    grid = create_grid_of_squares(square_edge_length,
                                  Point(-dome_circle_radius, -dome_circle_radius),
                                  Point(dome_circle_radius, dome_circle_radius))

    new_grid = []
    for row in grid:
        for square in row:
            if circ.contains_points(square.exterior_points()):
                new_grid.append(square)

    fig, ax = plt.subplots()
    circ.plot_to_ax(ax)
    plot_list_to_axis(grid, ax)
    plot_list_to_axis([h.inscribed_circle() for h in new_grid], ax)

    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.suptitle(f'{circ}, {len(new_grid)} squares of edge length {square_edge_length}')

    plt.show()

    circ = Circle(Point(0, 0), dome_circle_radius)
    grid = create_grid_of_hexagons(hexagon_circumradius,
                                   Point(-dome_circle_radius, -dome_circle_radius),
                                   Point(dome_circle_radius, dome_circle_radius))

    new_grid = []
    for row in grid:
        for square in row:
            if circ.contains_points(square.exterior_points()):
                new_grid.append(square)

    fig, ax = plt.subplots()
    circ.plot_to_ax(ax)
    plot_list_to_axis(grid, ax)
    plot_list_to_axis([h.inscribed_circle() for h in new_grid], ax)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    plt.show()

    """
    ++++++++++++++++
    THE ACTUAL CODE:
    +++++++++++++++
    """

    ###
    # Do everything
    inscribed_circle_radius = inter_cavern_spacing + salt_cavern_diameter / 2
    square_edge_length = inscribed_circle_radius * 2
    hexagon_circumradius = inscribed_circle_radius * 2 / math.sqrt(3)

    results = []

    for dome_circle_radius in range(3000, 20000, 500):
        circ = Circle(Point(0, 0), dome_circle_radius)
        square_grid = create_grid_of_squares(square_edge_length,
                                             Point(-dome_circle_radius, -dome_circle_radius),
                                             Point(dome_circle_radius, dome_circle_radius))
        square_grid = [square for row in square_grid for square in row]
        squares_inside = [s for s in square_grid if circ.contains_points(s.exterior_points())]
        hexagon_grid = create_grid_of_hexagons(hexagon_circumradius,
                                               Point(-dome_circle_radius, -dome_circle_radius),
                                               Point(dome_circle_radius, dome_circle_radius))
        hexagon_grid = [square for row in hexagon_grid for square in row]
        hexagons_inside = [s for s in hexagon_grid if circ.contains_points(s.exterior_points())]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        circ.plot_to_ax(ax1)
        circ.plot_to_ax(ax2)

        plot_list_to_axis(squares_inside, ax1)
        plot_list_to_axis([s.inscribed_circle() for s in squares_inside], ax1)

        plot_list_to_axis(hexagons_inside, ax2)
        plot_list_to_axis([s.inscribed_circle() for s in hexagons_inside], ax2)

        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')

        fig.tight_layout()
        fig.suptitle(f'Circle with radius {dome_circle_radius}\n '
                     f'{len(squares_inside)} squares inscribed circle radius {inscribed_circle_radius}\n'
                     f'{len(hexagons_inside)} hexagons with inscribed circle radius {inscribed_circle_radius} ')

        plt.savefig(f"Circle with radius {dome_circle_radius}.png")
        results.append({'dome_circle_radius': dome_circle_radius, 'squares': len(squares_inside),
                        'hexagons': len(hexagons_inside)})

    results = pd.DataFrame(results)
    fig, ax = plt.subplots()
    ax.plot(results['dome_circle_radius'], results['squares'], label='squares')
    ax.plot(results['dome_circle_radius'], results['hexagons'], label='hexagons')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"Squares and Hexagons.png")
