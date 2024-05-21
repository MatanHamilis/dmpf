use std::{borrow::BorrowMut, cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    utils::{BitSlice, BitVec},
    BITS_OF_SECURITY,
};

pub struct BinaryTrie {
    root: Rc<RefCell<TrieNode>>,
    len: usize,
}
impl Drop for BinaryTrie {
    fn drop(&mut self) {
        self.tear_down()
    }
}
impl BinaryTrie {
    fn tear_down(&mut self) {
        TrieNode::tear_down(self.root.deref());
    }
}
impl Default for BinaryTrie {
    fn default() -> Self {
        BinaryTrie {
            root: Rc::new(RefCell::new(TrieNode::default())),
            len: 0,
        }
    }
}
#[derive(Eq)]
pub struct TrieNode {
    is_terminal: bool,
    parent: Option<Rc<RefCell<TrieNode>>>,
    sons: [Option<Rc<RefCell<TrieNode>>>; 2],
}
impl PartialEq for TrieNode {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}
impl TrieNode {
    fn tear_down(node: &RefCell<Self>) {
        let left = node.borrow_mut().sons[0].take();
        let right = node.borrow_mut().sons[1].take();
        if let Some(l) = left {
            (&*l).borrow_mut().parent = None;
            TrieNode::tear_down(l.deref())
        }
        if let Some(r) = right {
            (&*r).borrow_mut().parent = None;
            TrieNode::tear_down(r.deref())
        }
    }
    pub fn set_terminal(&mut self, is_terminal: bool) {
        self.is_terminal = is_terminal;
    }
    pub fn connect(node: Rc<RefCell<TrieNode>>, son: Rc<RefCell<TrieNode>>, idx: bool) {
        (&*node).borrow_mut().sons[idx as usize] = Some(son.clone());
        (&*son).borrow_mut().parent = Some(node.clone());
    }
    pub fn get_son(&self, idx: bool) -> Option<Rc<RefCell<TrieNode>>> {
        self.sons[idx as usize].clone()
    }
    pub fn get_parent(&self) -> Option<Rc<RefCell<TrieNode>>> {
        self.parent.clone()
    }
}
impl Default for TrieNode {
    fn default() -> Self {
        Self {
            is_terminal: false,
            parent: None,
            sons: [None, None],
        }
    }
}
impl BinaryTrie {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn insert(&mut self, str: &BitSlice) {
        let mut cur_node = self.root.clone();
        for i in 0..str.len() {
            let bit = str.get(i);
            let son = cur_node.borrow().get_son(bit);
            if son.is_none() {
                let new_son = Rc::new(RefCell::new(TrieNode::default()));
                TrieNode::connect(cur_node.clone(), new_son, bit)
            }
            let new_node = cur_node.borrow().get_son(bit).unwrap();
            cur_node = new_node;
        }
        (&*cur_node).borrow_mut().set_terminal(true);
        self.len += 1;
    }
    pub fn iter_at_depth(&self, depth: usize) -> BinaryTrieDepthIter {
        BinaryTrieDepthIter::new(self, depth)
    }
}
pub struct BinaryTrieDepthIter<'a> {
    trie: &'a BinaryTrie,
    string: BitVec,
    depth: usize,
    cur_depth: usize,
    prev_item: Option<Rc<RefCell<TrieNode>>>,
    cur_item: Option<Rc<RefCell<TrieNode>>>,
}
impl<'a> BinaryTrieDepthIter<'a> {
    pub fn new(trie: &'a BinaryTrie, depth: usize) -> Self {
        Self {
            trie,
            string: BitVec::new(depth.max(1)),
            depth: depth + 1,
            cur_depth: 1,
            prev_item: None,
            cur_item: Some(trie.root.clone()),
        }
    }
    fn traverse_next(&mut self) {
        if self.cur_item.is_none() {
            return;
        }
        let cur_item = self.cur_item.as_ref().unwrap().clone();
        let prev_item = self.prev_item.clone();
        self.prev_item = self.cur_item.clone();
        if cur_item.borrow().parent == prev_item {
            let next_son = match cur_item.borrow().get_son(false) {
                Some(v) => Some((v, false)),
                None => cur_item.borrow().get_son(true).map(|v| (v, true)),
            };
            if next_son.is_some() && self.depth > self.cur_depth {
                let (next_son, direction) = next_son.unwrap();
                self.cur_item = Some(next_son);
                self.string.set(self.cur_depth - 1, direction);
                self.cur_depth += 1;
                return;
            }
            self.cur_item = cur_item.borrow().get_parent();
            self.cur_depth -= 1;
        } else if cur_item.borrow().get_son(false) == prev_item && prev_item.is_some() {
            let right_son = cur_item.borrow().get_son(true);
            if right_son.is_some() && self.depth > self.cur_depth {
                self.cur_item = right_son;
                self.string.set(self.cur_depth - 1, true);
                self.cur_depth += 1;
            } else {
                self.cur_item = cur_item.borrow().get_parent();
                self.cur_depth -= 1;
            }
        } else {
            self.cur_item = cur_item.borrow().get_parent();
            self.cur_depth -= 1;
        }
    }
    pub fn obtain_string(&self, output: &mut BitVec) {
        let num_cells = self.depth.div_ceil(BITS_OF_SECURITY);
        let last_cell_bits = self.depth & (BITS_OF_SECURITY - 1);
        self.string
            .as_ref()
            .iter()
            .zip(output.as_mut().iter_mut())
            .take(num_cells)
            .for_each(|(src, dst)| *dst = *src);
        output
            .as_mut()
            .get_mut(num_cells - 1)
            .iter_mut()
            .for_each(|v| v.mask(last_cell_bits));
    }
}
impl<'a> Iterator for BinaryTrieDepthIter<'a> {
    type Item = Rc<RefCell<TrieNode>>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.cur_depth == 0 {
                return None;
            }
            if self.cur_depth == self.depth {
                let output = self.cur_item.clone();
                self.traverse_next();
                return output;
            }
            self.traverse_next();
        }
    }
}

#[cfg(test)]
mod test {
    use super::BinaryTrie;
    use crate::utils::BitVec;

    #[test]
    fn trie_test() {
        let mut trie = BinaryTrie::default();
        let mut string = BitVec::new(1);
        trie.insert(&(&string).into());
        string.set(0, true);
        trie.insert(&(&string).into());
        for i in 0..1 {
            let t = trie.iter_at_depth(i);
            assert_eq!(t.count(), 1);
        }
        let t = trie.iter_at_depth(1);
        assert_eq!(t.count(), 2);
    }
}
