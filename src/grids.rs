use glam::{vec3, Vec3};
use std::ops::Range;

pub struct Grid<'a> {
    size: usize,
    pub vertices: &'a [Vec3],
    pub indices: &'a [u32],
}

impl<'a> Grid<'a> {
    pub fn new(size: usize, vertices: &'a [Vec3], indices: &'a [u32]) -> Self {
        Self { size, vertices, indices }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn triangles(&self) -> impl Iterator<Item = [Vec3; 3]> {
        let (chunks, _) = self.indices.as_chunks::<3>();

        chunks.iter().map( |chunk| {
            chunk.map(|i| self.vertices[i as usize])
        })
    }
}

pub struct Grids {
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
    vertices_starts: Vec<u32>,
    indices_starts: Vec<u32>,
    grid_sizes: Vec<u32>,
}

impl Grids {
    pub fn new(log2_range: Range<usize>) -> Self {
        let mut vertices =
            Vec::with_capacity(log2_range.clone().into_iter().map(|n| (n + 1).pow(2)).sum());
        let mut indices =
            Vec::with_capacity(log2_range.clone().into_iter().map(|n| (n + 1).pow(2)).sum());

        let mut vertices_starts = Vec::with_capacity(log2_range.len());
        let mut indices_starts = Vec::with_capacity(vertices_starts.capacity());

        for i in log2_range.clone() {
            let start_index = vertices.len() as u32;

            vertices_starts.push(start_index);
            indices_starts.push(indices.len() as u32);

            Self::build_grid(1 << i, &mut vertices, &mut indices, start_index);
        }

        Self {
            vertices,
            indices,
            vertices_starts,
            indices_starts,
            grid_sizes: log2_range.map(|n| 1 << n).collect(),
        }
    }

    pub fn build_grid(
        cell_count: u32,
        vertices: &mut Vec<Vec3>,
        indices: &mut Vec<u32>,
        start_index: u32,
    ) {
        let vert_count = cell_count + 1;

        for y in 0..vert_count {
            for x in 0..vert_count {
                vertices.push(vec3(
                    x as f32 / cell_count as f32,
                    y as f32 / cell_count as f32,
                    0.0,
                ));
            }
        }

        for y in 0..cell_count {
            for x in 0..cell_count {
                let tl = y * vert_count + x;
                indices.push(start_index + tl);
                indices.push(start_index + tl + 1);
                indices.push(start_index + tl + vert_count + 0);
                indices.push(start_index + tl + 1);
                indices.push(start_index + tl + vert_count + 1);
                indices.push(start_index + tl + vert_count + 0);
            }
        }
    }

    pub fn get_grid_sizes(&self) -> &[u32] {
        &self.grid_sizes[..]
    }

    fn get_grid_size_index(&self, size: usize) -> Option<usize> {
        self.grid_sizes.iter().position(|&n| n == size as u32)
    }

    fn get_nth_grid_start_indices(&self, n: usize) -> Option<(u32, u32)> {
        if n >= self.vertices_starts.len() {
            return None;
        }

        Some((self.vertices_starts[n], self.indices_starts[n]))
    }

    pub fn get_grid_size_start_indices(&self, size: usize) -> Option<(u32, u32)> {
        let i = self.get_grid_size_index(size)?;

        self.get_nth_grid_start_indices(i)
    }

    pub fn get_grid_size_index_ranges(&self, size: usize) -> Option<(Range<u32>, Range<u32>)> {
        let i = self.get_grid_size_index(size)?;

        let (verts_start, indices_start) = self.get_nth_grid_start_indices(i)?;
        let (verts_end, indices_end) = self
            .get_nth_grid_start_indices(i + 1)
            .unwrap_or((self.vertices.len() as u32, self.indices.len() as u32));

        Some((verts_start..verts_end, indices_start..indices_end))
    }

    pub fn get_grid_size_slices(&self, size: usize) -> Option<(&[Vec3], &[u32])> {
        let (vert_range, index_range) = self.get_grid_size_index_ranges(size)?;

        Some((
            &self.vertices[vert_range.start as usize..vert_range.end as usize],
            &self.indices[index_range.start as usize..index_range.end as usize],
        ))
    }

    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn get_grid(&self, size: usize) -> Option<Grid<'_>> {
        let (vertices, indices) = self.get_grid_size_slices(size)?;

        Some(Grid {
            size,
            vertices,
            indices,
        })
    }
}
