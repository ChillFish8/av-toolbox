use crate::frame::layers::{LayerId, Layers};

#[test]
fn a_missing_layer_is_chunk_zero() {
    let mut layers = Layers::default();
    layers.set(LayerId(1), 4);

    assert_eq!(layers.get(LayerId(0)), 0);
    assert_eq!(layers.get(LayerId(2)), 0);
}

#[test]
fn set_then_get_round_trips() {
    let mut layers = Layers::default();
    layers.set(LayerId(3), 7);
    layers.set(LayerId(0), 2);
    layers.set(LayerId(3), 9);

    assert_eq!(layers.get(LayerId(0)), 2);
    assert_eq!(layers.get(LayerId(3)), 9);

    let entries: Vec<(LayerId, u64)> = layers.iter().collect();
    assert_eq!(entries, vec![(LayerId(0), 2), (LayerId(3), 9)]);
}

#[test]
fn layers_compare_regardless_of_insertion_order() {
    let mut forwards = Layers::default();
    forwards.set(LayerId(0), 1);
    forwards.set(LayerId(1), 2);

    let mut backwards = Layers::default();
    backwards.set(LayerId(1), 2);
    backwards.set(LayerId(0), 1);

    assert_eq!(forwards, backwards);
}
