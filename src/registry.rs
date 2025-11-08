use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

#[derive(Eq, Hash, PartialEq)]
pub struct Id<T>(pub(super) usize, PhantomData<T>)
where
    T: Hash + Eq;

#[derive(Default)]
pub struct Registry<T: Hash + Eq> {
    next_id: usize,
    items: HashMap<usize, T>,
}

impl<T> Registry<T>
where
    T: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            next_id: 0,
            items: HashMap::new(),
        }
    }

    pub fn add(&mut self, item: T) -> Id<T> {
        let id = self.next_id;

        self.items.insert(id, item);

        self.next_id += 1;

        Id(id, PhantomData)
    }

    pub fn get(&self, id: &Id<T>) -> Option<&T> {
        self.items.get(&id.0)
    }

    pub fn get_mut(&mut self, id: &Id<T>) -> Option<&mut T> {
        self.items.get_mut(&id.0)
    }
}

impl<T> std::ops::Index<&Id<T>> for Registry<T>
where
    T: Hash + Eq,
{
    type Output = T;

    fn index(&self, index: &Id<T>) -> &Self::Output {
        self.get(index).unwrap()
    }
}
