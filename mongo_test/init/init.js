//db = db.getSiblingDB('mydatabase');
//db.createCollection('mycollection');
//db.mycollection.insert({ key: 'value' });

function getRandomCount() {
    return Math.floor(Math.random() * 20) + 1;
}

// Function to generate random data
function generateRandomData() {
    return {
        name: 'User' + Math.floor(Math.random() * 1000),
        age: Math.floor(Math.random() * 80) + 18,
        email: 'user' + Math.floor(Math.random() * 1000) + '@example.com',
        isActive: Math.random() > 0.5,
        createdAt: new Date(),
        score: Math.floor(Math.random() * 100)
    };
}

// Database 1: E-commerce
let db = db.getSiblingDB('ecommerce');

// Collections for ecommerce
db.createCollection('products');
db.createCollection('customers');
db.createCollection('orders');
db.createCollection('categories');
db.createCollection('reviews');

for (let i = 0; i < getRandomCount(); i++) {
    db.products.insert({
        name: 'Product' + i,
        price: Math.random() * 1000,
        stock: Math.floor(Math.random() * 100),
        category: 'Category' + Math.floor(Math.random() * 5)
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.customers.insert(generateRandomData());
}

for (let i = 0; i < getRandomCount(); i++) {
    db.orders.insert({
        orderId: 'ORD' + i,
        customerId: 'CUST' + Math.floor(Math.random() * 100),
        total: Math.random() * 5000,
        status: ['pending', 'completed', 'shipped'][Math.floor(Math.random() * 3)]
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.categories.insert({
        name: 'Category' + i,
        description: 'Description for category ' + i,
        isActive: Math.random() > 0.5
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.reviews.insert({
        productId: 'PROD' + Math.floor(Math.random() * 100),
        rating: Math.floor(Math.random() * 5) + 1,
        comment: 'Review comment ' + i
    });
}

// Database 2: Blog
db = db.getSiblingDB('blog');

// Collections for blog
db.createCollection('posts');
db.createCollection('comments');
db.createCollection('users');
db.createCollection('tags');
db.createCollection('categories');

for (let i = 0; i < getRandomCount(); i++) {
    db.posts.insert({
        title: 'Blog Post ' + i,
        content: 'Content for post ' + i,
        author: 'Author' + Math.floor(Math.random() * 10),
        views: Math.floor(Math.random() * 1000)
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.comments.insert({
        postId: 'POST' + Math.floor(Math.random() * 100),
        content: 'Comment ' + i,
        author: 'User' + Math.floor(Math.random() * 100)
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.users.insert(generateRandomData());
}

for (let i = 0; i < getRandomCount(); i++) {
    db.tags.insert({
        name: 'Tag' + i,
        usageCount: Math.floor(Math.random() * 100)
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.categories.insert({
        name: 'BlogCategory' + i,
        description: 'Blog category description ' + i
    });
}

// Database 3: Analytics
db = db.getSiblingDB('analytics');

// Collections for analytics
db.createCollection('pageViews');
db.createCollection('userSessions');
db.createCollection('events');
db.createCollection('errors');
db.createCollection('performance');

for (let i = 0; i < getRandomCount(); i++) {
    db.pageViews.insert({
        url: '/page' + Math.floor(Math.random() * 10),
        timestamp: new Date(),
        userId: 'USER' + Math.floor(Math.random() * 100),
        duration: Math.random() * 300
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.userSessions.insert({
        sessionId: 'SESSION' + i,
        userId: 'USER' + Math.floor(Math.random() * 100),
        startTime: new Date(),
        duration: Math.random() * 3600
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.events.insert({
        type: ['click', 'scroll', 'hover'][Math.floor(Math.random() * 3)],
        element: 'element' + i,
        timestamp: new Date()
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.errors.insert({
        code: Math.floor(Math.random() * 500),
        message: 'Error message ' + i,
        timestamp: new Date(),
        severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
    });
}

for (let i = 0; i < getRandomCount(); i++) {
    db.performance.insert({
        metric: 'metric' + i,
        value: Math.random() * 100,
        timestamp: new Date()
    });
}
