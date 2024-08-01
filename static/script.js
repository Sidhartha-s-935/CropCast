
function getTableData() {
    const crops = [];
    const productions = [];
    const table = document.querySelector('table');
    const rows = table.querySelectorAll('tr');

    // Loop through the first five rows, skipping the header row
    for (let i = 1; i <= 5; i++) {
        const cells = rows[i].querySelectorAll('td');
        crops.push(cells[0].innerText);
        productions.push(parseFloat(cells[2].innerText));
    }
    return { crops, productions };
}

// Create the pie chart
function createPieChart(crops, productions) {
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
            title: {
                display: true,
                text: 'Top 5 Crop Production (1000 tons)'
            }
        }
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

    // Initialize Select2
    stateSelect.select2({
        placeholder: 'Select State',
        allowClear: true
    });

    districtSelect.select2({
        placeholder: 'Select District',
        allowClear: true
    });

    stateSelect.on('change', function () {
        const selectedState = $(this).val();
        const districts = stateToDistricts[selectedState] || [];

        // Create new options for districts
        const districtOptions = districts.map(district => new Option(district, district, false, false));

        // Update district select
        districtSelect.empty().append(districtOptions).trigger('change');
    });

    // Ensure default options are set on page load
    stateSelect.val(null).trigger('change');
    districtSelect.empty().append('<option value="" disabled selected>Select District</option>').trigger('change');

    const { crops, productions } = getTableData();
    createPieChart(crops, productions);
});