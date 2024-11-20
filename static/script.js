document.addEventListener('DOMContentLoaded', function () {
    const { crops: pieCrops, productions: pieProductions } = getTableData(5, 'production');
    createPieChart(pieCrops, pieProductions);
    populatePieChartExplanation(pieCrops, pieProductions);  // Populate explanation for pie chart

    const { crops: barCrops, productions: barYields } = getTableData(10, 'yield');
    createBarChart(barCrops, barYields);

    // Sort bar chart data by production in ascending order
    const sortedBarData = sortDataByProduction(barCrops, barYields);
    populateBarChartExplanation(sortedBarData.crops, sortedBarData.productions);  // Populate explanation for bar chart
});

// Function to sort crops and productions based on production value in ascending order
function sortDataByProduction(crops, productions) {
    const data = crops.map((crop, index) => {
        return { crop, production: productions[index] };
    });

    // Sort by production value in ascending order
    data.sort((a, b) => b.production - a.production);

    // Unzip the sorted data into separate arrays
    const sortedCrops = data.map(item => item.crop);
    const sortedProductions = data.map(item => item.production);

    return { crops: sortedCrops, productions: sortedProductions };
}

function getTableData(limit, type) {
    const crops = [];
    const values = [];
    const table = document.querySelector('table');
    const rows = table.querySelectorAll('tr');
    for (let i = 1; i <= limit && i < rows.length; i++) {
        const cells = rows[i].querySelectorAll('td');
        crops.push(cells[0].innerText);
        if (type === 'production') {
            values.push(parseFloat(cells[2].innerText));
        } else if (type === 'yield') {
            values.push(parseFloat(cells[3].innerText));
        }
    }
    return { crops, productions: values };
}

function createPieChart(crops, productions, titleColor = '#000') {
    const ctx = document.getElementById('myPieChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: crops,
            datasets: [{
                data: productions,
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF'],
                hoverBackgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'High-Production Crops',
                    font: {
                        size: 32,
                        family: 'Jost',
                        weight: 'lighter'
                    },
                    color: titleColor // Allows customization of the title color
                },
                legend: {
                    display: true,
                    position: 'bottom',
                    align: 'center'
                },
                tooltip: {
                    callbacks: {
                        label: function (tooltipItem) {
                            const value = tooltipItem.raw;
                            return `${value} Ton(s)`;
                        }
                    }
                }
            }
        }
    });
}


function populatePieChartExplanation(crops, productions) {
    const explanationBox = document.getElementById('pieExplanationBox');
    explanationBox.innerHTML = '';  // Clear any existing explanation

    crops.forEach((crop, index) => {
        const production = productions[index];
        const explanation = document.createElement('p');
        explanation.textContent = `${index + 1}. ${crop}: ${production} ton(s)`;
        explanationBox.appendChild(explanation);
    });
}

function createBarChart(crops, productions, titleColor = '#000') {
    const ctx = document.getElementById('myBarChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: crops,
            datasets: [{
                data: productions,
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'],
                hoverBackgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'High Yield Crops',
                    font: {
                        size: 32,
                        family: 'Jost',
                        weight: 'lighter'
                    },
                    color: titleColor // Allows customization of the title color
                },
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        label: function (tooltipItem) {
                            const value = tooltipItem.raw;
                            return `${value} Ton(s) per acres`;
                        }
                    }
                }
            }
        }
    });
}

function populateBarChartExplanation(crops, productions) {
    const explanationBox = document.getElementById('barExplanationBox');
    explanationBox.innerHTML = '';  // Clear any existing explanation

    crops.forEach((crop, index) => {
        const production = productions[index];
        const explanation = document.createElement('p');
        explanation.textContent = `${index + 1}. ${crop}: ${production} tons per acres`;
        explanationBox.appendChild(explanation);
    });
}

document.addEventListener("DOMContentLoaded", function () {
    const stateToDistricts = {
        "Andhra Pradesh": ['Ananthapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa YSR', 'Krishna', 'Kurnool', 'S.P.S. Nellore', 'Srikakulam', 'Visakhapatnam', 'West Godavari'],
        "Assam": ['Cachar', 'Darrang', 'Dibrugarh', 'Goalpara', 'Kamrup', 'Karbi Anglong', 'Lakhimpur', 'Nagaon', 'North Cachar Hil / Dima hasao', 'Sibsagar'],
        "Bihar": ['Bhagalpur', 'Champaran', 'Darbhanga', 'Gaya', 'Mungair', 'Muzaffarpur', 'Patna', 'Purnea', 'Saharsa', 'Saran', 'Shahabad (now part of Bhojpur district)'],
        "Chhattisgarh": ['Bastar', 'Bilaspur', 'Durg', 'Raigarh', 'Raipur', 'Surguja'],
        "Gujarat": ['Ahmedabad', 'Amreli', 'Banaskantha', 'Bharuch', 'Bhavnagar', 'Dangs', 'Jamnagar', 'Junagadh', 'Kheda', 'Kutch', 'Mehsana', 'Panchmahal', 'Rajkot', 'Sabarkantha', 'Surat', 'Surendranagar', 'Vadodara / Baroda', 'Valsad'],
        "Haryana": ['Ambala', 'Gurgaon', 'Hissar', 'Jind', 'Karnal', 'Mahendragarh / Narnaul', 'Rohtak'],
        "Himachal Pradesh": ['Bilashpur', 'Chamba', 'Kangra', 'Kinnaur', 'Kullu', 'Lahul & Spiti', 'Mandi', 'Shimla', 'Sirmaur', 'Solan'],
        "Jharkhand": ['Dhanbad', 'Hazaribagh', 'Palamau', 'Ranchi', 'Santhal Paragana / Dumka', 'Singhbhum'],
        "Karnataka": ['Bangalore', 'Belgaum', 'Bellary', 'Bidar', 'Bijapur / Vijayapura', 'Chickmagalur', 'Chitradurga', 'Dakshina Kannada', 'Dharwad', 'Gulbarga / Kalaburagi', 'Hassan', 'Kodagu / Coorg', 'Kolar', 'Mandya', 'Mysore', 'Raichur', 'Shimoge', 'Tumkur', 'Uttara Kannada'],
        "Kerala": ['Alappuzha', 'Eranakulam', 'Kannur', 'Kollam', 'Kottayam', 'Kozhikode', 'Malappuram', 'Palakkad', 'Thiruvananthapuram', 'Thrissur'],
        "Madhya Pradesh": ['Balaghat', 'Betul', 'Bhind', 'Chhatarpur', 'Chhindwara', 'Damoh', 'Datia', 'Dewas', 'Dhar', 'Guna', 'Gwalior', 'Hoshangabad', 'Indore', 'Jabalpur', 'Jhabua', 'Khandwa / East Nimar', 'Khargone / West Nimar', 'Mandla', 'Mandsaur', 'Morena', 'Narsinghpur', 'Panna', 'Raisen', 'Rajgarh', 'Ratlam', 'Rewa', 'Sagar', 'Satna', 'Sehore', 'Seoni / Shivani', 'Shahdol', 'Shajapur', 'Shivpuri', 'Sidhi', 'Tikamgarh', 'Ujjain', 'Vidisha'],
        "Maharashtra": ['Ahmednagar', 'Akola', 'Amarawati', 'Aurangabad', 'Beed', 'Bhandara', 'Bombay', 'Buldhana', 'Chandrapur', 'Dhule', 'Jalgaon', 'Kolhapur', 'Nagpur', 'Nanded', 'Nasik', 'Osmanabad', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Solapur', 'Thane', 'Wardha', 'Yeotmal'],
        "Orissa": ['Balasore', 'Bolangir', 'Cuttack', 'Dhenkanal', 'Ganjam', 'Kalahandi', 'Keonjhar', 'Koraput', 'Mayurbhanja', 'Phulbani ( Kandhamal )', 'Puri', 'Sambalpur', 'Sundargarh'],
        "Punjab": ['Amritsar', 'Bhatinda', 'Ferozpur', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar', 'Kapurthala', 'Ludhiana', 'Patiala', 'Roopnagar / Ropar', 'Sangrur'],
        "Rajasthan": ['Ajmer', 'Alwar', 'Banswara', 'Barmer', 'Bharatpur', 'Bhilwara', 'Bikaner', 'Bundi', 'Chittorgarh', 'Churu', 'Dungarpur', 'Ganganagar', 'Jaipur', 'Jaisalmer', 'Jalore', 'Jhalawar', 'Jhunjhunu', 'Jodhpur', 'Kota', 'Nagaur', 'Pali', 'Sikar', 'Sirohi', 'Swami Madhopur', 'Tonk', 'Udaipur'],
        "Tamilnadu": ['Chengalpattu MGR / Kanchipuram', 'Coimbatore', 'Kanyakumari', 'Madurai', 'North Arcot / Vellore', 'Ramananthapuram', 'Salem', 'South Arcot / Cuddalore', 'Thanjavur', 'The Nilgiris', 'Thirunelveli', 'Tiruchirapalli / Trichy'],
        "Telangana": ['Adilabad', 'Hyderabad', 'Karimnagar', 'Khammam', 'Mahabubnagar', 'Medak', 'Nalgonda', 'Nizamabad', 'Warangal'],
        "Uttar Pradesh": ['Agra', 'Aligarh', 'Allahabad', 'Azamgarh', 'Bahraich', 'Ballia', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bijnor', 'Budaun', 'Buland Shahar', 'Deoria', 'Etah', 'Etawah', 'Faizabad', 'Farrukhabad', 'Fatehpur', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hardoi', 'Jalaun', 'Jaunpur', 'Jhansi', 'Kanpur', 'Kheri', 'Lucknow', 'Mainpuri', 'Mathura', 'Meerut', 'Mirzpur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh', 'Rae-Bareily', 'Rampur', 'Saharanpur', 'Shahjahanpur', 'Sitapur', 'Sultanpur', 'Unnao', 'Varanasi'],
        "Uttarakhand": ['Almorah', 'Chamoli', 'Dehradun', 'Garhwal', 'Nainital', 'Pithorgarh', 'Tehri Garhwal', 'Uttar Kashi'],
        "West Bengal": ['24 Parganas', 'Bankura', 'Birbhum', 'Burdwan', 'Cooch Behar', 'Darjeeling', 'Hooghly', 'Howrah', 'Jalpaiguri', 'Malda', 'Midnapur', 'Murshidabad', 'Nadia', 'Purulia', 'West Dinajpur'],
        };
    const stateSelect = $('select[name="option1"]');
    const districtSelect = $('select[name="option2"]');

    stateSelect.on('change', function () {
        const selectedState = $(this).val();
        const districts = stateToDistricts[selectedState] || [];
        const districtOptions = districts.map(district => new Option(district, district, false, false));
        districtSelect.empty().append(districtOptions).trigger('change');
    });
    
    stateSelect.val(null).trigger('change');
    districtSelect.empty().append('<option value="" disabled selected>Select District</option>').trigger('change');

    const { crops, productions } = getTableData();
    createPieChart(crops, productions);
});
