extends ProgressBar

@export var player: Player;

func _ready() -> void:
    max_value = 20

func _process(delta: float) -> void:
    var tween: Tween = get_tree().create_tween()
    tween.tween_property(self, "value", player.health, 0.2)
