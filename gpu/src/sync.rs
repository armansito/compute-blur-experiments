use std::sync::{Arc, Condvar, Mutex};

#[derive(Clone)]
pub struct Semaphore {
    data: Arc<(Mutex<u32>, Condvar)>,
}

impl Semaphore {
    pub fn new(capacity: u32) -> Self {
        Self {
            data: Arc::new((Mutex::new(capacity), Condvar::new())),
        }
    }

    pub fn acquire(&self) {
        let mut value = self.data.0.lock().unwrap();
        while *value == 0 {
            value = self.data.1.wait(value).unwrap();
        }
        *value -= 1;
    }

    pub fn release(&self) {
        let mut value = self.data.0.lock().unwrap();
        *value += 1;
        self.data.1.notify_one();
    }
}
