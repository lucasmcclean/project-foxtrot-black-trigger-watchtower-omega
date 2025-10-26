extends ProgressBar

@export var player: Player;

func _ready():
	max_value = 20

func _process(delta: float) -> void:
	value = player.health
