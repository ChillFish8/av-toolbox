use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::frame::gpu::FrameMeta;
use crate::frame::layers::LayerId;
use crate::frame::provided::{ChunkRoute, Provided, ProvidedValue};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Flag(bool);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Chunk(u64);

fn hash_of(provided: &Provided) -> u64 {
    let mut hasher = DefaultHasher::new();
    provided.hash(&mut hasher);
    hasher.finish()
}

fn read_chunk(value: &dyn ProvidedValue) -> u64 {
    value.as_any().downcast_ref::<Chunk>().unwrap().0
}

#[test]
fn a_value_reads_back_as_its_own_type() {
    let mut provided = Provided::new();
    provided.set(LayerId(1), Flag(true));

    assert_eq!(provided.get::<Flag>(LayerId(1)), Some(&Flag(true)));
}

#[test]
fn a_value_reads_back_as_none_as_another_type() {
    let mut provided = Provided::new();
    provided.set(LayerId(1), Flag(true));

    assert_eq!(provided.get::<Chunk>(LayerId(1)), None);
}

#[test]
fn a_missing_layer_reads_as_none() {
    let mut provided = Provided::new();
    provided.set(LayerId(1), Flag(true));

    assert_eq!(provided.get::<Flag>(LayerId(0)), None);
    assert!(provided.get_raw(LayerId(2)).is_none());
}

#[test]
fn stores_set_in_any_order_compare_and_hash_equal() {
    let mut forwards = Provided::new();
    forwards.set(LayerId(0), Flag(true));
    forwards.set(LayerId(2), Chunk(5));

    let mut backwards = Provided::new();
    backwards.set(LayerId(2), Chunk(5));
    backwards.set(LayerId(0), Flag(true));

    assert_eq!(forwards, backwards);
    assert_eq!(hash_of(&forwards), hash_of(&backwards));
}

#[test]
fn stores_with_different_values_differ() {
    let mut on = Provided::new();
    on.set(LayerId(0), Flag(true));

    let mut off = Provided::new();
    off.set(LayerId(0), Flag(false));

    assert_ne!(on, off);
}

#[test]
fn a_clone_does_not_see_later_writes() {
    let mut original = Provided::new();
    original.set(LayerId(0), Flag(true));

    let copy = original.clone();
    original.set(LayerId(0), Flag(false));
    original.set(LayerId(1), Chunk(2));

    assert_eq!(copy.get::<Flag>(LayerId(0)), Some(&Flag(true)));
    assert_eq!(copy.get::<Chunk>(LayerId(1)), None);
    assert_eq!(copy.layers().count(), 1);
}

#[test]
fn a_route_reads_its_layer_through_its_key() {
    let route = ChunkRoute {
        layer: LayerId(3),
        key: read_chunk,
    };
    let meta = FrameMeta::at(0).with_provided(LayerId(3), Chunk(7));

    assert_eq!(route.chunk_of(&meta), 7);
}

#[test]
fn a_route_reads_chunk_zero_without_a_value() {
    let route = ChunkRoute {
        layer: LayerId(3),
        key: read_chunk,
    };

    assert_eq!(route.chunk_of(&FrameMeta::at(0)), 0);
}
