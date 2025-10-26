extends ProgressBar

@export var player: Player
@onready var particles: GPUParticles2D = $GlowParticle

var last_health: int

func _ready() -> void:
	max_value = 20
	value = player.health
	last_health = value

func _process(delta: float) -> void:
	if player.health < last_health:
		particles.restart()
		particles.emitting = true

	var tween: Tween = get_tree().create_tween()
	tween.tween_property(self, "value", player.health, 0.2)

	last_health = player.health
