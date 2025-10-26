extends Parallax2D

@export var speeds := {
	"Sky": 5.0,
	"Sun": 10.0,
	"Clouds": 20.0,
	"Mountains": 10.0
}

var offsets := {}

func _ready():
	for child in get_children():
		offsets[child.name] = Vector2.ZERO

func _process(delta):
	for child in get_children():
		var spd = speeds.get(child.name, 0.0)
		offsets[child.name].x += spd * delta

		# Make sure the texture scrolls forever
		var region = child.region_rect
		region.position.x = fmod(offsets[child.name].x, region.size.x)
		child.region_rect = region
